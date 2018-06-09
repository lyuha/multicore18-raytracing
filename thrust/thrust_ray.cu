#include <iostream>
#include <ctime>
#include <cstdint>
#include <cmath>
#include <memory.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/pair.h>

#define rnd( x ) (x * rand() / RAND_MAX)
#define SPHERES 20
#define INF 2e10f
#define DIM 2048

struct Sphere {
	float r, g, b;
	float radius;
	float x, y, z;

	__host__ __device__
	Sphere() : r(0), g(0), b(0), radius(0), x(0), y(0), z(0) {}

	__host__ __device__
	Sphere(const Sphere& s) {
		this->r = s.r;
		this->g = s.g;
		this->b = s.b;
		this->radius = s.radius;
		this->x = s.x;
		this->y = s.y;
		this->z = s.z;
	}

	__device__
	float hit(const float& ox, const float& oy, float& n) const {
		float dx = ox - x;
		float dy = oy - y;
		if (dx * dx + dy * dy < radius * radius) {
			float dz = std::sqrt(radius * radius - dx * dx - dy * dy);
			n = dz / radius;
			return dz + z;
		}
		return -INF;
	}
};

__device__
float is_hit(const Sphere& sphere, const float& ox, const float& oy, float& n) {
	float dx = ox - sphere.x;
	float dy = oy - sphere.y;
	if (dx * dx + dy * dy < sphere.radius * sphere.radius) {
		float dz = std::sqrt(sphere.radius * sphere.radius - dx * dx - dy * dy);
		n = dz / std::sqrt(sphere.radius * sphere.radius);
		return dz + sphere.z;
	}
	return -INF;
}

__host__
Sphere generate_random_sphere() {
	Sphere s = Sphere();
	s.r = rnd(1.0f);
	s.g = rnd(1.0f);
	s.b = rnd(1.0f);
	s.x = rnd(2000.0f) - 1000;
	s.y = rnd(2000.0f) - 1000;
	s.z = rnd(2000.0f) - 1000;
	s.radius = rnd(200.0f) + 40;

	return s;
}

struct Color {
	uint8_t r, g, b, a;

	__host__ __device__
	Color() : r(0), g(0), b(0), a(255) {}

	__host__ __device__
	Color(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a = 255) : r(_r), g(_g), b(_b), a(_a) {}

	__host__ __device__
	Color(const Color& c) {
		this->r = c.r;
		this->g = c.g;
		this->b = c.b;
		this->a = c.a;
	}
};

struct color_from_sphere : public thrust::unary_function<const Sphere&, thrust::pair<Color, float>> {
	int ox;
	int oy;

	__host__ __device__
	color_from_sphere(const int& _ox, const int& _oy) : ox(_ox), oy(_oy) {}

	__device__
	thrust::pair<Color, float> operator() (const Sphere& s) {
		float fscale;
		float t = s.hit(ox, oy, fscale);

		Color color(0, 0, 0, 255);

		if (t > -INF) {
			color.r = (uint8_t)(s.r * fscale * 255);
			color.g = (uint8_t)(s.g * fscale * 255);
			color.b = (uint8_t)(s.b * fscale * 255);
		}

		return thrust::make_pair<Color, float>(color, t);
	}
};

struct get_color_max_distance : public thrust::binary_function<thrust::pair<Color, float>, thrust::pair<Color, float>, thrust::pair<Color, float>> {
	__device__
	thrust::pair<Color, float> operator() (thrust::pair<Color, float> const& lhs, thrust::pair<Color, float> const& rhs) {
		if (lhs.second > rhs.second) {
			return lhs;
		} else {
			return rhs;
		}
	}
};

struct determine_color_functor : public thrust::unary_function<const int&, Color> {
	thrust::device_vector<Sphere>::iterator start;
	thrust::device_vector<Sphere>::iterator end;

	determine_color_functor(thrust::device_vector<Sphere>::iterator _start,
		thrust::device_vector<Sphere>::iterator _end) : start(_start), end(_end) {}

	__device__
	Color operator() (const int& offset) const {
		int x = offset % DIM;
		int y = (offset - x) / DIM;

		float ox = x - DIM / 2.0;
		float oy = y - DIM / 2.0;

		thrust::pair<Color, float> ray_color;

		ray_color.first = Color(0, 0, 0, 255);
		ray_color.second = -INF;

		// Spheres -> transform to Color, distance(T) -> reduce to Color
		thrust::pair<Color, float> result = thrust::transform_reduce(thrust::device, start, end, color_from_sphere(ox, oy), ray_color, get_color_max_distance());

		return result.first;
	}
};

void write_ppm(thrust::host_vector<Color> const& bitmap, const int& xdim, const int& ydim, FILE* fp)
{
	fprintf(fp, "P3\n");
	fprintf(fp, "%d %d\n", xdim, ydim);
	fprintf(fp, "255\n");
	for (int y = 0; y < ydim; ++y) {
		for (int x = 0; x < xdim; ++x) {
			int i = x + y * xdim;
			fprintf(fp, "%d %d %d ", bitmap[i].r, bitmap[i].g, bitmap[i].b);
		}
		fprintf(fp, "\n");
	}
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		exit(-1);
	}

	FILE* fp = fopen(argv[1], "w");

	thrust::host_vector<Color> image(DIM * DIM);
	thrust::host_vector<Sphere> spheres(SPHERES);
	thrust::generate(spheres.begin(), spheres.end(), generate_random_sphere);

	thrust::device_vector<Sphere> device_spheres = spheres;
	thrust::device_vector<int> offset_equence(DIM * DIM);
	thrust::device_vector<Color> bitmap(image.size());
	thrust::sequence(offset_equence.begin(), offset_equence.end());

	clock_t startTime = clock();

	thrust::transform(offset_equence.cbegin(), offset_equence.cend(), bitmap.begin(), determine_color_functor(device_spheres.begin(), device_spheres.end()));
	cudaThreadSynchronize();

	clock_t endTime = clock();

	write_ppm(bitmap, DIM, DIM, fp);

	fclose(fp);

	double execute_time = (double)(endTime - startTime) / CLOCKS_PER_SEC;

	std::cout << "Thrust ray tracing: " << std::fixed << execute_time << " sec" << std::endl;

	return 0;
}
