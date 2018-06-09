#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <time.h>

#define rnd( x ) (x * rand() / RAND_MAX)
#define SPHERES 20
#define INF 2e10f
#define DIM 2048

struct Sphere {
	float r, b, g;
	float radius;
	float x, y, z;

	__device__ float hit(float ox, float oy, float *n) {
		float dx = ox - x;
		float dy = oy - y;
		if (dx * dx + dy * dy < radius * radius) {
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};

void ppm_write(unsigned char* bitmap, int xdim, int ydim, FILE* fp)
{
	fprintf(fp, "P3\n");
	fprintf(fp, "%d %d\n", xdim, ydim);
	fprintf(fp, "255\n");
	for (int y = 0; y < ydim; y++) {
		for (int x = 0; x < xdim; x++) {
			int i = x + y * xdim;
			fprintf(fp, "%d %d %d ", bitmap[4 * i], bitmap[4 * i + 1], bitmap[4 * i + 2]);
		}
		fprintf(fp, "\n");
	}
}

__global__ void kernel(Sphere *s, unsigned char *ptr) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = x - DIM / 2;
	float oy = y - DIM - 2;
	float r = 0, g = 0, b = 0;
	float maxz = -INF;
	for (int i = 0; i < SPHERES; i++) {
		float n;
		float t = s[i].hit(ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}
	}
	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;
}

int main(int argc, char *argv[]) {
	unsigned char *dev_bitmap, *bitmap;
	FILE* fp = fopen(argv[1], "w");
	Sphere *s;

	cudaMalloc((void **)&dev_bitmap, sizeof(unsigned char) * DIM * DIM * 4);
	cudaMalloc((void **)&s, sizeof(Sphere) * SPHERES);

	Sphere *temp_s = new Sphere[SPHERES];
	bitmap = (unsigned char *)malloc(sizeof(unsigned char) * DIM * DIM * 4);
	//bitmap 생성하고 다시 하자.
	for (int i = 0; i < SPHERES; i++) {
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(2000.0f) - 1000;
		temp_s[i].y = rnd(2000.0f) - 1000;
		temp_s[i].z = rnd(2000.0f) - 1000;
		temp_s[i].radius = rnd(200.0f) + 40;
	}
	cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);
	delete[] temp_s;

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);

	clock_t startTime = clock();
	kernel<<<grids, threads>>>(s, dev_bitmap);
	cudaDeviceSynchronize();
	clock_t endTime = clock();

	cudaMemcpy(bitmap, dev_bitmap, sizeof(unsigned char) * DIM * DIM * 4, cudaMemcpyDeviceToHost);

	ppm_write(bitmap, DIM, DIM, fp);

	cudaFree(dev_bitmap);
	cudaFree(s);
	fclose(fp);

	double execute_time = (double)(endTime - startTime) / CLOCKS_PER_SEC;

	std::cout << "CUDA ray tracing: " << std::fixed << execute_time << " sec" << std::endl;
	return 0;
}
