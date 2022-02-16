#include <iostream>
#include "BlackJack.h"
using namespace std;

int main()
{
	// how much play
	int play;
	cout << "Play Game : ";
	cin >> play;

	// play
	BlackJack blackJack;
	blackJack.play(play);

	return 0;
}