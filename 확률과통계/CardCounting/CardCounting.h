#pragma once

class CardCounting
{
private:
	int cardList[11];	// list of remaining card
	int myCard[11];		// player's card
	int myA;			// player's A number
	double dealerBust;	// probability of dealer bust

public:
	CardCounting();

	// setter
	void setCardList();
	void setMyCard(int card[]);
	void setA(int a);
	void setDealerBust(int card);

	// delete used card
	void usedCard(int card);
	
	// get probability of dealer bust
	double getDealerBust(double sum, int total);

	// Hit or Stand
	bool HitStatus();
};