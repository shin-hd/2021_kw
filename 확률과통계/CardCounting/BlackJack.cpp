#include <iostream>
#include <cstdlib>
#include <ctime>
#include "BlackJack.h"
using namespace std;

BlackJack::BlackJack()
{
	cardPoint = 0;
	dealer = new Player;
	player1 = new Player;
	player2 = new Player;
	CC = new CardCounting;
}

BlackJack::~BlackJack()
{
	delete dealer;
	delete player1;
	delete player2;
}

void BlackJack::shuffle()
{
	// deck reset
	cardPoint = 0;
	for (int i = 0; i < 6 * 4; i++)
	{
		for (int j = 0; j < 13; j++)
		{
			deck[i * 13 + j] = j + 1;
		}
	}

	// shuffle
	srand((unsigned int)time(NULL));
	for (int i = 6*4*13 - 1; i > 0; i--)
	{
		int randNum = rand() % (i + 1);
		int temp = deck[i];
		deck[i] = deck[randNum];
		deck[randNum] = deck[i];
	}
}

void BlackJack::play(int play)
{
	cardPoint = 0;
	shuffle();
	CC->setCardList();

	for (int i = 0; i < play; i++)
	{
		// 80% of cards are used, shuffle
		if (cardPoint > 0.8 * 6 * 4 * 13 - 1)
		{
			cardPoint = 0;
			shuffle();
			CC->setCardList();
		}

		// initialize players
		dealer->initCard();
		player1->initCard();
		player2->initCard();

		// hidden card
		int hidden = cardPoint;
		dealer->pushCard(deck[cardPoint++]);
		
		player1->pushCard(deck[cardPoint]);
		CC->usedCard(deck[cardPoint++]);
		player2->pushCard(deck[cardPoint]);
		CC->usedCard(deck[cardPoint++]);
		
		// reveal one dealer card
		dealer->pushCard(deck[cardPoint]);
		CC->usedCard(deck[cardPoint]);
		CC->setDealerBust(deck[cardPoint++]);

		player1->pushCard(deck[cardPoint]);
		CC->usedCard(deck[cardPoint++]);
		player2->pushCard(deck[cardPoint]);
		CC->usedCard(deck[cardPoint++]);

		// set card, a
		CC->setMyCard(player1->getCard());
		CC->setA(player1->getA());

		// betting
		player1->betting();
		player2->betting();

		// check blackjack
		int p1bj = 0, p2bj = 0;
		if (player1->SumofNum() == 21)
			p1bj = 1;
		if (player2->SumofNum() == 21)
			p2bj = 1;

		// play player
		while (CC->HitStatus() || (player2->SumofNum() < 17))
		{
			if (CC->HitStatus())
			{
				player1->pushCard(deck[cardPoint]);
				CC->usedCard(deck[cardPoint++]);
				CC->setMyCard(player1->getCard());
				CC->setA(player1->getA());
			}
			if (player2->SumofNum() < 17)
			{
				player2->pushCard(deck[cardPoint]);
				CC->usedCard(deck[cardPoint++]);
			}
		}

		// open hidden card
		CC->usedCard(deck[hidden]);
		// play dealer
		while (dealer->SumofNum() < 17)
		{
			dealer->pushCard(deck[cardPoint]);
			CC->usedCard(deck[cardPoint++]);
		}

		// blackjack
		if (p1bj)
		{
			player1->calcBlackjack();
			player1->isWin();
		}
		else if (!player1->isBurst())
		{
			// dealer is loser
			if (dealer->isBurst() || (player1->SumofNum() > dealer->SumofNum()))
			{
				player1->calcWin();
				player1->isWin();
			}
			// draw
			else if(player1->SumofNum() == dealer->SumofNum())
				player1->calcDraw();
		}

		// blackjack
		if (p2bj)
		{
			player2->calcBlackjack();
			player2->isWin();
		}
		else if (!player2->isBurst())
		{
			// win
			if (dealer->isBurst() || (player2->SumofNum() > dealer->SumofNum()))
			{
				player2->calcWin();
				player2->isWin();
			}
			// draw
			else if (player2->SumofNum() == dealer->SumofNum())
				player2->calcDraw();
		}
	}
	// print result
	printResult(play);
}

void BlackJack::printResult(int play)
{
	// player1 info
	cout << "<< Player1 >>" << endl;
	cout << "½Â¸® : " << player1->getWin() << "\t¹«½ÂºÎ ¹× ÆÐ¹è : " << play - player1->getWin() << endl;
	cout << "money : " << player1->getMoney() << endl;
	
	cout << "\n";
	
	// player2 info
	cout << "<< Player2 >>" << endl;
	cout << "½Â¸® : " << player2->getWin() << "\t¹«½ÂºÎ ¹× ÆÐ¹è : " << play - player2->getWin() << endl;
	cout << "money : " << player2->getMoney() << endl;
}
