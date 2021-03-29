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

int qFunc(int i, int j, int k, const int N, int**** colors)
{
	for (int c = 0; c < 3; ++c)
		if (colors[k][c][i][j] != 0)
			return 0;

	bool all = true;
	for (int k_ = 0; k_ < N; ++k_)
		for (int c = 0; c < 3; ++c)
			if (colors[k_][c][i][j] != 0)
				all = false;

	if (all)
		return 0;
	else
		return 1000000;
}

float norm(int k, int k_, int i, int j, int**** colors)
{
	return pow(pow(colors[k][0][i][j] - colors[k_][0][i][j], 2) +
		       pow(colors[k][1][i][j] - colors[k_][1][i][j], 2) +
		       pow(colors[k][2][i][j] - colors[k_][2][i][j], 2), 0.5);
}

float g(int t, int t_, int k, int k_, const int width, int**** colors)
{
	return norm(k, k_, t / width, t % width, colors) + norm(k, k_, t_ / width, t_ % width, colors);
}

int** run(const int height, const int width, const int modKs, int*** q, int**** colors, const int loops)
{
	// Initialize phi, L, R, U, D
	int modT = height * width;
	float** phi = new float* [modT];
	float** L = new float* [modT];
	float** R = new float* [modT];
	float** U = new float* [modT];
	float** D = new float* [modT];
	for (int ij = 0; ij < modT; ++ij)
	{
		phi[ij] = new float[modKs]();
		L[ij] = new float[modKs]();
		R[ij] = new float[modKs]();
		U[ij] = new float[modKs]();
		D[ij] = new float[modKs]();
	}

	for (int i = height - 2; i >= 0; --i)
	{
		const int i_ = i * width;
		for (int j = width - 2; j >= 0; --j)
		{
			const int ij = i_ + j;
			for (int k = 0; k < modKs; ++k)
			{
				float minR = 100000.;
				for (int k_ = 0; k_ < modKs; ++k_)
				{
					const float R_ = R[ij + 1][k_] + 0.5 * q[i][j + 1][k_] + g(ij + 1, ij, k_, k, width, colors);
					if (R_ < minR)
						minR = R_;
				}
				R[i_ + j][k] = minR;

				float minD = 100000.;
				for (int k_ = 0; k_ < modKs; ++k_)
				{
					const float D_ = D[ij + width][k_] + 0.5 * q[i + 1][j][k_] + g(ij + width, ij, k_, k, width, colors);
					if (D_ < minD)
						minD = D_;
				}
				D[i_ + j][k] = minD;
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
					float minL = 100000.;
					for (int k_ = 0; k_ < modKs; ++k_)
					{
						const float L_ = L[ij - 1][k_] + 0.5 * q[i][j - 1][k_] + g(ij - 1, ij, k_, k, width, colors) - phi[ij - 1][k_];
						if (L_ < minL)
							minL = L_;
					}
					L[ij][k] = minL;

					float minU = 100000.;
					for (int k_ = 0; k_ < modKs; ++k_)
					{
						const float U_ = U[ij - width][k_] + 0.5 * q[i - 1][j][k_] + g(ij - width, ij, k_, k, width, colors) + phi[ij - width][k_];
						if (U_ < minU)
							minU = U_;
					}
					U[ij][k] = minU;

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
					float minR = 100000.;
					for (int k_ = 0; k_ < modKs; ++k_)
					{
						const float R_ = R[ij + 1][k_] + 0.5 * q[i][j + 1][k_] + g(ij + 1, ij, k_, k, width, colors) - phi[ij + 1][k_];
						if (R_ < minR)
							minR = R_;
					}
					R[ij][k] = minR;

					float minD = 100000.;
					for (int k_ = 0; k_ < modKs; ++k_)
					{
						const float D_ = D[ij + width][k_] + 0.5 * q[i + 1][j][k_] + g(ij + width, ij, k_, k, width, colors) + phi[ij + width][k_];
						if (D_ < minD)
							minD = D_;
					}
					D[ij][k] = minD;

					phi[ij][k] = (L[ij][k] + R[ij][k] - U[ij][k] - D[ij][k]) * 0.5;
				}
			}
		}
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time used for " << loops << " iterations : " << float(duration.count()) / 1000000. << endl;

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
			float value = 100000.;
			for (int k_ = 0; k_ < modKs; ++k_)
			{
				const float v_ = L[ij][k_] + R[ij][k_] + 0.5 * q[i][j][k_] - phi[ij][k_];
				if (v_ < value)
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
	int**** colors = new int*** [N];
	for (int k = 0; k < N; ++k)
	{
		vector<Mat> channels(3);
		Mat image;
		image = imread("res" + to_string(k + 1) + ".png", IMREAD_UNCHANGED);
		split(image, channels);
		height = channels[0].size().height;
		width = channels[0].size().width;
		colors[k] = new int** [3];
		for (int c = 0; c < 3; ++c)
		{
			colors[k][c] = new int* [height];
			for (int i = 0; i < height; ++i)
			{
				colors[k][c][i] = new int[width];
				for (int j = 0; j < width; ++j)
				{
					colors[k][c][i][j] = int(channels[c].at<uchar>(i, j));
				}
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

	const int loops = 10;
	int** res = run(height, width, modKs, q, colors, loops);

	//then merge them back
	Mat result, chanels[3];
	for (int c = 0; c < 3; ++c)
	{
		chanels[c] = Mat::zeros(Size(width, height), CV_8UC1);
		for (int i = 0; i < height; ++i)
			for (int j = 0; j < width; ++j)
				chanels[c].at<uchar>(i, j) = uchar(colors[res[i][j]][c][i][j]);
	}

	merge(chanels, 3, result);

	namedWindow("Result image", WINDOW_AUTOSIZE);
	imshow("Result image", result);
	imwrite("result.png", result);

	waitKey(0);
	return 0;
}