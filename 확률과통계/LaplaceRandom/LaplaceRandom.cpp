#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
using namespace std;

double laplaceRandom(double m, double b);

int main()
{
	int m, b;
	int n;

	cout << "평균: ";
	cin >> m;
	cout << "스케일: ";
	cin >> b;
	cout << "난수 수: ";
	cin >> n;

	ofstream write;
	write.open("laplaceRandom.txt");

	if (write.is_open())
	{
		// generate n laplace random numbers
		for (int i = 0; i < n; i++)
			write << laplaceRandom(m, b) << endl;
	}
}

double laplaceRandom(double m, double b) {
	double v = 0;

	v = (double)rand() / RAND_MAX; // 0 ~ 1 uniform random numbbers

	if (v > 0.5)	// v > 0.5
		return m - b * log(2.0 - (2 * v));	// T(v)
	else		// v <= 0.5
		return m + b * log(2 * v);			// T(v)
}