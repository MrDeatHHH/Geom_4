#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <chrono> 
#include <cmath>
#include <cassert>
#include <string>

using namespace cv;
using namespace std;
using namespace std::chrono;

int qFunc(int i, int j, int k, const int N, int*** colors)
{
	if (colors[k][i][j] != 0)
		return 0;
	else
	{
		bool all = true;
		for (int k_ = 0; k_ < N; ++k_)
			if (colors[k_][i][j] != 0)
				all = false;

		if (all)
			return 0;
		else
			return -100000000;
	}
}

int g(int t, int t_, int k, int k_, const int width, int*** colors)
{
	return abs(colors[k][t_ / width][t_ % width] - colors[k_][t_ / width][t_ % width]) - abs(colors[k][t / width][t % width] - colors[k_][t / width][t % width]);
}

int** run(const int height, const int width, const int modKs, int*** q, int*** colors, const int loops)
{
	// Initialize phi, L, R, U, D
	int modT = height * width;
	double** phi = new double* [modT];
	double** L = new double* [modT];
	double** R = new double* [modT];
	double** U = new double* [modT];
	double** D = new double* [modT];
	for (int ij = 0; ij < modT; ++ij)
	{
		phi[ij] = new double[modKs]();
		L[ij] = new double[modKs]();
		R[ij] = new double[modKs]();
		U[ij] = new double[modKs]();
		D[ij] = new double[modKs]();
	}

	for (int i = height - 2; i >= 0; --i)
	{
		const int i_ = i * width;
		for (int j = width - 2; j >= 0; --j)
		{
			const int ij = i_ + j;
			for (int k = 0; k < modKs; ++k)
			{
				double maxR = -10000000.;
				for (int k_ = 0; k_ < modKs; ++k_)
				{
					const double R_ = R[ij + 1][k_] + 0.5 * q[i][j + 1][k_] + g(ij + 1, ij, k_, k, width, colors);
					if (R_ > maxR)
						maxR = R_;
				}
				R[i_ + j][k] = maxR;

				double maxD = -10000000.;
				for (int k_ = 0; k_ < modKs; ++k_)
				{
					const double D_ = D[ij + width][k_] + 0.5 * q[i + 1][j][k_] + g(ij + width, ij, k_, k, width, colors);
					if (D_ > maxD)
						maxD = D_;
				}
				D[i_ + j][k] = maxD;
			}
		}
	}

	auto start = high_resolution_clock::now();
	// Main loop
	for (int iter = 0; iter < loops; ++iter)
	{
		// Forward
		for (int i = 1; i < height; ++i)
		{
			const int i_ = i * width;
			for (int j = 1; j < width; ++j)
			{
				const int ij = i_ + j;
				for (int k = 0; k < modKs; ++k)
				{
					double maxL = -10000000.;
					for (int k_ = 0; k_ < modKs; ++k_)
					{
						const double L_ = L[ij - 1][k_] + 0.5 * q[i][j - 1][k_] + g(ij - 1, ij, k_, k, width, colors) - phi[ij - 1][k_];
						if (L_ > maxL)
							maxL = L_;
					}
					L[ij][k] = maxL;

					double maxU = -10000000.;
					for (int k_ = 0; k_ < modKs; ++k_)
					{
						const double U_ = U[ij - width][k_] + 0.5 * q[i - 1][j][k_] + g(ij - width, ij, k_, k, width, colors) + phi[ij - width][k_];
						if (U_ > maxU)
							maxU = U_;
					}
					U[ij][k] = maxU;

					phi[ij][k] = (L[ij][k] + R[ij][k] - U[ij][k] - D[ij][k]) * 0.5;
				}
			}
		}
		// Backward
		for (int i = height - 2; i >= 0; --i)
		{
			const int i_ = i * width;
			for (int j = width - 2; j >= 0; --j)
			{
				const int ij = i_ + j;
				for (int k = 0; k < modKs; ++k)
				{
					double maxR = -10000000.;
					for (int k_ = 0; k_ < modKs; ++k_)
					{
						const double R_ = R[ij + 1][k_] + 0.5 * q[i][j + 1][k_] + g(ij + 1, ij, k_, k, width, colors) - phi[ij + 1][k_];
						if (R_ > maxR)
							maxR = R_;
					}
					R[ij][k] = maxR;

					double maxD = -10000000.;
					for (int k_ = 0; k_ < modKs; ++k_)
					{
						const double D_ = D[ij + width][k_] + 0.5 * q[i + 1][j][k_] + g(ij + width, ij, k_, k, width, colors) + phi[ij + width][k_];
						if (D_ > maxD)
							maxD = D_;
					}
					D[ij][k] = maxD;

					phi[ij][k] = (L[ij][k] + R[ij][k] - U[ij][k] - D[ij][k]) * 0.5;
				}
			}
		}
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time used for " << loops << " iterations : " << double(duration.count()) / 1000000. << endl;

	// Best Ks
	int** res = new int* [height];
	for (int i = 0; i < height; ++i)
	{
		res[i] = new int[width]();
	}

	for (int i = 0; i < height; ++i)
	{
		const int i_ = i * width;
		for (int j = 0; j < width; ++j)
		{
			const int ij = i_ + j;
			int k_star = 0;
			double value = -10000000.;
			for (int k_ = 0; k_ < modKs; ++k_)
			{
				const double v_ = L[ij][k_] + R[ij][k_] + 0.5 * q[i][j][k_] - phi[ij][k_];
				if (v_ > value)
				{
					value = v_;
					k_star = k_;
				}
			}
			res[i][j] = k_star;
		}
	}

	for (int ij = 0; ij < modT; ++ij)
	{
		delete[] phi[ij];
		delete[] L[ij];
		delete[] R[ij];
		delete[] U[ij];
		delete[] D[ij];
	}
	delete[] phi;
	delete[] L;
	delete[] R;
	delete[] U;
	delete[] D;

	return res;
}

int main()
{
	const int N = 3;

	int height = 0;
	int width = 0;

	// Get array from Mat
	int*** colors = new int** [N];
	for (int k = 0; k < N; ++k)
	{
		Mat image_, image;
		image_ = imread("res" + to_string(k + 1) + ".png", IMREAD_UNCHANGED);
		cvtColor(image_, image, COLOR_BGR2GRAY);
		height = image.size().height;
		width = image.size().width;
		colors[k] = new int* [height];
		for (int i = 0; i < height; ++i)
		{
			colors[k][i] = new int[width];
			for (int j = 0; j < width; ++j)
			{
				colors[k][i][j] = int(image.at<uchar>(i, j));
			}
		}
	}

	// Target colors
	const int epsilon = 0;
	const int modKs = N;
	const int modT = height * width;
	
	// Q
	int*** q = new int** [height];
	for (int i = 0; i < height; ++i)
	{
		q[i] = new int* [width];
		for (int j = 0; j < width; ++j)
		{
			q[i][j] = new int[modKs];
			for (int k = 0; k < modKs; ++k)
				q[i][j][k] = qFunc(i, j, k, N, colors);
		}
	}

	const int loops = 30;
	int** res = run(height, width, modKs, q, colors, loops);

	Mat result = Mat::zeros(Size(width, height), CV_8UC1);
	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
			result.at<uchar>(i, j) = uchar(colors[res[i][j]][i][j]);

	namedWindow("Result image", WINDOW_AUTOSIZE);
	imshow("Result image", result);
	imwrite("result.png", result);

	waitKey(0);
	return 0;
}