#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define CUDA 0
#define OPENMP 1
#define SPHERES 20

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF 2e10f
#define DIM 2048

typedef struct Sphere {
	float   r, b, g;
	float   radius;
	float   x, y, z;
} Sphere;

float hit(Sphere* s, float ox, float oy, float *n) {
	float dx = ox - s->x;
	float dy = oy - s->y;
	if (dx*dx + dy * dy < s->radius*s->radius) {
		float dz = sqrtf(s->radius*s->radius - dx * dx - dy * dy);
		*n = dz / s->radius;
		return dz + s->z;
	}
	return -INF;
}

void kernel(int x, int y, Sphere* s, unsigned char* ptr)
{
	int offset = x + y * DIM;
	float ox = (x - DIM / 2.0);
	float oy = (y - DIM / 2.0);

	//printf("x:%d, y:%d, ox:%f, oy:%f\n",x,y,ox,oy);

	float r = 0, g = 0, b = 0;
	float maxz = -INF;

	for (int i = 0; i < SPHERES; i++) {
		float n;
		float t = hit(&s[i], ox, oy, &n);
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

void ppm_write(unsigned char* bitmap, const int xdim, const int ydim, FILE* fp)
{
	int i, x, y;
	fprintf(fp, "P3\n");
	fprintf(fp, "%d %d\n", xdim, ydim);
	fprintf(fp, "255\n");
	for (y = 0; y < ydim; y++) {
		for (x = 0; x < xdim; x++) {
			i = x + y * xdim;
			fprintf(fp, "%d %d %d ", bitmap[4 * i], bitmap[4 * i + 1], bitmap[4 * i + 2]);
		}
		fprintf(fp, "\n");
	}
}

int main(int argc, char* argv[])
{
	int no_threads;

	int x, y;
	unsigned char* bitmap;
	FILE *fp;

	srand((unsigned int)time(NULL));

	if (argc != 3) {
		printf("> a.out [option] [filename.ppm]\n");
		printf("[option] 1~16: OpenMP using 1~16 threads\n");
		printf("for example, '> a.out 8 result.ppm' means executing OpenMP with 8 threads\n");
		exit(0);
	}

	fopen_s(&fp, argv[2], "w");

	no_threads = atoi(argv[1]);

	omp_set_num_threads(no_threads);

	Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);

	int i = 0;

	for (i = 0; i < SPHERES; i++) {
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(2000.0f) - 1000;
		temp_s[i].y = rnd(2000.0f) - 1000;
		temp_s[i].z = rnd(2000.0f) - 1000;
		temp_s[i].radius = rnd(200.0f) + 40;
	}

	bitmap = (unsigned char*)malloc(sizeof(unsigned char)*DIM*DIM * 4);

	clock_t start_clock = clock();

#pragma omp parallel for schedule(static) default(none) private(x, y) shared(bitmap, temp_s)
	for (x = 0; x < DIM; x++) {
		for (y = 0; y < DIM; y++) {
			kernel(x, y, temp_s, bitmap);
		}
	}

	clock_t end_clock = clock();

	ppm_write(bitmap, DIM, DIM, fp);

	fclose(fp);
	free(bitmap);
	free(temp_s);

	double execute_time = ((double)(end_clock - start_clock)) / CLOCKS_PER_SEC;

	printf_s("OpenMP (%d threads) ray tracing: %f sec", no_threads, execute_time);

	return 0;
}
