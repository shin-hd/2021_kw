#pragma once
#include <iostream>
#include "CardCounting.h"

class Player
{
private:
	int card[11];	// max card num: 2*11 = 22
	int a;			// 'A' count
	int win;		// win count
	int money;		// money

public:
	Player()
	{
		initCard();
		win = 0;
		money = 100000;
	}

	// initialize
	void initCard()
	{
		for (int i = 0; i < 11; i++)
			card[i] = 0;
		a = 0;
	}

	// getter
	int* getCard() { return card; }
	int getA() { return a; }
	int getWin() { return win; }
	int getMoney() { return money; }

	// get sum of card number
	int SumofNum() {
		// get sum excluding A
		int sum = 0;
		for (int i = 0; i < 11; i++) {
			if (card[i] != 1)
				sum = sum + card[i];
		}

		// predict a to 1
		sum += a;		
		
		// if a can be 11 point card
		for (int i = a; i > 0; i--) {
			if (sum + 10 <= 21)		sum += 10;
		}
		return sum;
	}

	// setter
	void setCard(int* newCardList) {
		for (int i = 0; i < 11; i++) {
			card[i] = newCardList[i];
		}
	}
	void setMoney(int newMoney) { money = newMoney; }

	// push new card
	void pushCard(int newCard) {
		// A count ++
		if (newCard == 1)	a++;
		// push new card
		else
		{
			for (int i = 0; i < 11; i++)
				if (card[i] == 0)
				{
					if (newCard < 10)	card[i] = newCard;
					else				card[i] = 10;
					return;
				}
		}
	}
	// winner
	void isWin() { win = win + 1; }
	// burst check
	bool isBurst() {
		if (SumofNum() > 21)
			return true;
		return false;
	}
	// betting
	void betting() { money -= 100; }
	// win
	void calcWin() { money += 200; }
	// draw
	void calcDraw() { money += 100; }
	// black jack
	void calcBlackjack() { money += 250; }
};

class BlackJack
{
private:
	int deck[6 * 4 * 13];	// 4*13 each card count
	int cardPoint;		// used card pointer
	Player* dealer;
	Player* player1;	// card counting
	Player* player2;
	CardCounting* CC;

public:
	BlackJack();
	~BlackJack();

	// shuffle deck
	void shuffle();

	// play game
	void play(int play);
	
	// print result
	void printResult(int play);
};