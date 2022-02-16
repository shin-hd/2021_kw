#include <iostream>
#include "CardCounting.h"
using namespace std;

CardCounting::CardCounting()
{
	setCardList();
	myCard[0] = 0;
	myA = 0;
	dealerBust = 0;
}

void CardCounting::setCardList()
{
	// cardList[0] is not used
	cardList[0] = 0;
	
	// cardList[n] = n point's card count
	for (int i = 1; i < 10; i++)
		cardList[i] = 6 * 4;
	cardList[10] = 6 * 4 * 4;
}

void CardCounting::setMyCard(int card[])
{
	for(int i = 0; i < 11; i++)
		myCard[i] = card[i];
}

void CardCounting::setA(int a) { myA = a; }

void CardCounting::setDealerBust(int card)
{
	// get total card num
	int total = 0;
	for (int i = 1; i < 11; i++)
		total += cardList[i];

	// predict hidden card except A
	double sum = 0;
	for (int i = 2; i < 11; i++)
		sum += (i * (double)cardList[i]);
	sum /= total;	// n * p(n)
	if (card < 10)
		sum += card;	// n point
	else
		sum += 10;		// 10 point

	dealerBust = getDealerBust(sum, total);
}

void CardCounting::usedCard(int card)
{
		// if card is 10KQJ
		if (card >= 10 && cardList[10] != 0)
			cardList[10] -= 1;
		else if (cardList[card] != 0)
			cardList[card] -= 1;
}

double CardCounting::getDealerBust(double sum, int total)
{
	// dealer stand
	if (sum > 16)
		return 0;

	// get probability of bust
	double pBust = 0;
	for (int i = 0; i < 11; i++)
	{
		if (sum + i > 21)
			pBust += cardList[i];
	}
	pBust /= total;

	// predict next card
	for (int i = 2; i < 11; i++)
		sum += ((double)i * cardList[i]) / total;

	// P(sum) + (1 - P(sum)) * P(new sum)
	return pBust + (1-pBust)*getDealerBust(sum, total);
}

bool CardCounting::HitStatus()
{
	// if dealer may bust, stand
	if (dealerBust > 0.9)
		return false;

	// get total card num
	int total = 0;
	for (int i = 1; i < 11; i++)
		total += cardList[i];

	// get mySum excluding A
	int mySum = 0;
	for (int i = 0; i < 11; i++)
		mySum += myCard[i];

	// get my sum
	mySum += myA * 11;
	for (int i = myA; i > 0; i--)
	{
		// if my sum is sufficiently large value
		if (mySum > 17 && mySum < 22) break;
		// 11 to 1
		else	mySum -= 10;
	}

	// get probability of player bust
	double pBust = 0;
	for (int i = 0; i < 11; i++)
	{
		if (mySum + i > 21)
			pBust += cardList[i];
	}
	pBust /= total;

	// if bust prob is low, Hit	// if dealer bust prob is low, Hit
	if (pBust < 0.2 || (pBust < 0.4 && dealerBust < 0.3))
		return true;

	// if bust prob is high, Stand
	return false;
}