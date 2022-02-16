#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
using namespace std;

class Node
{
private:
	double p;
	char ascii;
	Node* leftNode;
	Node* rightNode;
public:
	Node(double p)
		:p(p), ascii(NULL), leftNode(NULL), rightNode(NULL) {}
	Node(double p, char ascii)
		:p(p), ascii(ascii), leftNode(NULL), rightNode(NULL) {}
	~Node() { delete leftNode; delete rightNode; }

	void setLeft(Node* left) { leftNode = left; }
	void setRight(Node* right) { rightNode = right; }

	double getP() { return p; }
	char getASCII() { return ascii; }
	Node* getLeft() { return leftNode; }
	Node* getRight() { return rightNode; }

	bool hasLeft() {
		if (leftNode == NULL) return false;
		return true;
	}
	bool hasRight() {
		if (rightNode == NULL) return false;
		return true;
	}
};

class HuffmanEncoder
{
private:
	int fileSize;
	char* fileData;
	double ASCII_probability[128];
	Node* huffmanTree;
	pair<int, unsigned short> huffmanTable[129];
public:
	HuffmanEncoder();
	~HuffmanEncoder();

	// read file of fileName
	bool readFile(string fileName);

	// get ASCII probability
	bool getProbability();

	// get Huffman Tree
	bool getHuffmanTree();

	// get Huffman Table
	bool getHuffmanTable();

	// huffman tree travelsal
	void traversal(Node*, int, unsigned short);

	// store Huffman Code
	bool storeHuffmanCode();
	// store Huffman Table
	bool storeHuffmanTable();

	// Do Encoding
	void encoding();

	// print huffman table
	void printHuffmanTable();
};

HuffmanEncoder::HuffmanEncoder()
{
	fileSize = 0;
	fileData = NULL;
	for (int i = 0; i < 128; i++)
	{
		ASCII_probability[i] = 0;
		huffmanTable[i] = pair<int, unsigned short>(0, NULL);
	}
	huffmanTable[128] = pair<int, unsigned short>(0, NULL);
	huffmanTree = NULL;
}

HuffmanEncoder::~HuffmanEncoder()
{
	delete huffmanTree;
}

bool HuffmanEncoder::readFile(string fileName)
{
	ifstream read;
	read.open(fileName);

	// file is not opened
	if (read.fail())
	{
		return false;
	}

	// read file data
	string stringData = "";
	char ch;
	while (!read.eof())
	{
		read.get(ch);
		stringData += ch;
	}
	read.close();

	// string to char array
	fileSize = stringData.length() + 1;
	fileData = new char[fileSize];
	strcpy_s(fileData, fileSize, stringData.c_str());

	return true;
}

bool HuffmanEncoder::getProbability()
{
	// get frequency
	for (int i = 0; i < fileSize; i++)
	{
		ASCII_probability[fileData[i]]++;
	}

	// get probability
	for (int i = 0; i < 128; i++)
	{
		ASCII_probability[i] /= fileSize;
	}

	return true;
}

bool compare(pair<double, Node*> a, pair<double, Node*> b)
{// descending
	return a.first > b.first;
}

bool HuffmanEncoder::getHuffmanTree()
{
	vector<pair<double, Node*>> list;
	Node* temp;
	for (int i = 0; i < 128; i++)
	{
		// probability should not be 0
		if (ASCII_probability[i] != 0)
		{ // insert
			temp = new Node(ASCII_probability[i], (char)i);
			list.push_back(pair<double, Node*>(ASCII_probability[i], temp));
		}
	}
	// insert EOD
	temp = new Node(1 / fileSize, 0);
	list.push_back(pair<double, Node*>(0, temp));

	// list is not empty && list.front probability is not 1
	while (!list.empty() && list.front().first < 1)
	{
		// sort list
		sort(list.begin(), list.end(), compare);

		// reverse iterator
		vector<pair<double, Node*>>::reverse_iterator rit = list.rbegin();
		double newP = rit->first;
		Node* firstNode = rit->second;
		rit++;

		// create new node
		newP += rit->first;
		temp = new Node(newP);
		temp->setLeft(firstNode);
		temp->setRight(rit->second);

		// delete 2 pair of end
		list.pop_back();
		list.pop_back();
		// push new pair
		list.push_back(pair<double, Node*>(newP, temp));
	}
	// last temp is root
	huffmanTree = temp;

	return true;
}

void HuffmanEncoder::traversal(Node* node, int num, unsigned short bit)
{
	// not root, no children
	if (!node->hasLeft() && !node->hasRight() && num != 0)
	{
		// [char]
		if (node->getASCII() != NULL)
			huffmanTable[node->getASCII()] = pair<int, unsigned short>(num, bit);
		// EOD
		else
			huffmanTable[128] = pair<int, unsigned short>(num, bit);
	}
	// has child
	if (node->hasLeft())
		traversal(node->getLeft(), num + 1, bit << 1);
	if (node->hasRight())
		traversal(node->getRight(), num + 1, (bit << 1) + 1);
}

bool HuffmanEncoder::getHuffmanTable()
{
	// don't have tree
	if (huffmanTree == NULL)
		return false;

	traversal(huffmanTree, 0, 0);
	return true;
}

bool HuffmanEncoder::storeHuffmanCode()
{
	ofstream write;
	write.open("huffman_code.hbs", ios::out | ios::binary);

	// file open error
	if (write.fail())
		return false;

	// init all
	int codeCount = 0;
	int usedBit = 0;

	unsigned char* code = new unsigned char[fileSize];
	for (int i = 0; i < fileSize; i++)
		code[i] = 0;

	unsigned char temp1 = 0, temp2 = 0;

	// get code of file data
	for (int i = 0; i < fileSize; i++)
	{
		usedBit %= 8;
		char ch = fileData[i];
		unsigned short huffmanCode = huffmanTable[ch].second;
		// bit size < 8
		if (huffmanTable[ch].first < 8)
			huffmanCode = huffmanCode << (8 - huffmanTable[ch].first); // left align

		// can use bit > huffman code bit size
		if (huffmanTable[ch].first < 8 - usedBit)
		{
			// bit
			temp1 |= huffmanCode >> usedBit;
			usedBit += huffmanTable[ch].first;
		}
		// can use bit + 1 byte > huffman code bit size
		else if (huffmanTable[ch].first < 16 - usedBit)
		{
			// bit size > 8
			if (huffmanTable[ch].first > 8)
			{
				temp1 |= huffmanCode >> (huffmanTable[ch].first % 8 + usedBit);
				temp2 |= huffmanCode << 8 - (huffmanTable[ch].first % 8 + usedBit);
			}
			// bit size <= 8
			else
			{
				temp1 |= huffmanCode >> usedBit;
				temp2 |= huffmanCode << 8 - usedBit;
			}
			usedBit += huffmanTable[ch].first;

			// insert and init
			code[codeCount] = temp1;
			codeCount += 1;
			temp1 = temp2;
			temp2 = 0;
		}
		else
		{
			// bit
			temp1 |= huffmanCode >> (huffmanTable[ch].first % 8 + usedBit);
			temp2 |= huffmanCode >> (huffmanTable[ch].first % 8 + usedBit - 8);

			// insert and init
			code[codeCount] = temp1;
			codeCount += 1;
			code[codeCount] = temp2;
			codeCount += 1;
			temp1 = 0;
			temp2 = 0;

			// bit
			temp1 |= huffmanCode >> (huffmanTable[ch].first % 8 + usedBit - 8);
			usedBit += huffmanTable[ch].first;
		}
	}

	usedBit %= 8;
	unsigned short huffmanCode = huffmanTable[128].second;
	// bit size < 8
	if (huffmanTable[128].first < 8)
		huffmanCode = huffmanCode << (8 - huffmanTable[128].first); // left align

	// can use bit > huffman code bit size
	if (huffmanTable[128].first < 8 - usedBit)
	{
		// bit
		temp1 |= huffmanCode >> usedBit;
		usedBit += huffmanTable[128].first;
	}
	// can use bit + 1 byte > huffman code bit size
	else if (huffmanTable[128].first < 16 - usedBit)
	{
		// bit size > 8
		if (huffmanTable[128].first > 8)
		{
			temp1 |= huffmanCode >> (huffmanTable[128].first % 8 + usedBit);
			temp2 |= huffmanCode << 8 - (huffmanTable[128].first % 8 + usedBit);
		}
		// bit size <= 8
		else
		{
			temp1 |= huffmanCode >> usedBit;
			temp2 |= huffmanCode << 8 - usedBit;
		}
		usedBit += huffmanTable[128].first;

		// insert and init
		code[codeCount] = temp1;
		codeCount += 1;
		temp1 = temp2;
		temp2 = 0;
	}
	else
	{
		// bit
		temp1 |= huffmanCode >> (huffmanTable[128].first % 8 + usedBit);
		temp2 |= huffmanCode >> (huffmanTable[128].first % 8 + usedBit - 8);

		// insert and init
		code[codeCount] = temp1;
		codeCount += 1;
		code[codeCount] = temp2;
		codeCount += 1;
		temp1 = 0;
		temp2 = 0;

		// bit
		temp1 |= huffmanCode >> (huffmanTable[128].first % 8 + usedBit - 8);
		usedBit += huffmanTable[128].first;
	}

	// write
	for (int i = 0; i < codeCount; i++)
	{
		write.write((char*)&code[i], 1);
	}

	delete[] code;
	write.close();
	return true;
}

bool HuffmanEncoder::storeHuffmanTable()
{
	ofstream write;
	write.open("huffman_table.hbs", ios::out | ios::binary);

	if (write.fail())
		return false;

	int codeCount = 0;
	int usedBit = 0;
	unsigned char temp1 = 0, temp2 = 0;
	unsigned char code[1000] = { 0. };

	// get table code
	for (int i = 0; i < 128; i++)
	{
		// not used ascii
		if (huffmanTable[i].first == 0) continue;

		// ascii
		usedBit %= 8;
		temp1 |= (unsigned char)i >> usedBit;
		temp2 |= (unsigned char)i << (8 - usedBit);

		// insert and init
		code[codeCount] = temp1;
		codeCount += 1;
		temp1 = temp2;
		temp2 = 0;

		// bit size
		temp1 |= (unsigned char)huffmanTable[i].first >> usedBit;
		temp2 |= (unsigned char)huffmanTable[i].first << (8 - usedBit);

		// insert and init
		code[codeCount] = temp1;
		codeCount += 1;
		temp1 = temp2;
		temp2 = 0;

		unsigned short huffmanCode = huffmanTable[i].second;
		// bit size < 8
		if (huffmanTable[i].first < 8)
			huffmanCode = huffmanCode << (8 - huffmanTable[i].first); // left align

		// can use bit > huffman code bit size
		if (huffmanTable[i].first < 8 - usedBit)
		{
			// bit
			temp1 |= huffmanCode >> usedBit;
			usedBit += huffmanTable[i].first;
		}
		// can use bit + 1 byte > huffman code bit size
		else if (huffmanTable[i].first < 16 - usedBit)
		{
			// bit size > 8
			if (huffmanTable[i].first > 8)
			{
				temp1 |= huffmanCode >> (huffmanTable[i].first % 8 + usedBit);
				temp2 |= huffmanCode << 8 - (huffmanTable[i].first % 8 + usedBit);
			}
			// bit size <= 8
			else
			{
				temp1 |= huffmanCode >> usedBit;
				temp2 |= huffmanCode << 8 - usedBit;
			}
			usedBit += huffmanTable[i].first;

			// insert and init
			code[codeCount] = temp1;
			codeCount += 1;
			temp1 = temp2;
			temp2 = 0;
		}
		else
		{
			// bit
			temp1 |= huffmanCode >> (huffmanTable[i].first % 8 + usedBit);
			temp2 |= huffmanCode >> (huffmanTable[i].first % 8 + usedBit - 8);

			// insert and init
			code[codeCount] = temp1;
			codeCount += 1;
			code[codeCount] = temp2;
			codeCount += 1;
			temp1 = 0;
			temp2 = 0;

			// bit
			temp1 |= huffmanCode >> (huffmanTable[i].first % 8 + usedBit - 8);
			usedBit += huffmanTable[i].first;
		}

	}

	// write
	for (int i = 0; i < codeCount; i++)
	{
		write.write((char*)&code[i], 1);
	}

	write.close();
	return true;
}

void HuffmanEncoder::encoding()
{
	if (!getProbability()) {
		cout << "getProbability: Error" << endl;
		return;
	}
	if (!getHuffmanTree()) {
		cout << "getHuffmanTree: Error" << endl;
		return;
	}
	if (!getHuffmanTable()) {
		cout << "getHuffmanTable: Error" << endl;
		return;
	}
	if (!storeHuffmanTable()) {
		cout << "storeHuffmanTable: Error" << endl;
		return;
	}
	if (!storeHuffmanCode()) {
		cout << "storeHuffmanCode: Error" << endl;
		return;
	}
}
void HuffmanEncoder::printHuffmanTable()
{
	for (int i = 0; i < 128; i++)
	{
		unsigned short p = 1;
		if (huffmanTable[i].first != 0) {
			printf("%x", i);
			cout << '\t' << huffmanTable[i].first << '\t' << (int)huffmanTable[i].second << '\t';
			p = p << huffmanTable[i].first - 1;
			for (int j = 0; j < huffmanTable[i].first; j++) {
				if (huffmanTable[i].second & p)
					cout << 1;
				else
					cout << 0;
				p = p >> 1;
			}
			cout << '\n';
		}
	}
}
/////////////////////////////
// 1. 파일 열어서 읽기
// 2. ASCII 확률 구하기
// 3. Huffman Tree
// 4. Table 변환.
//		Table은 int(bit 길이)와 unsigned char(bit pattern)의 pair array[129]
// 5. 파일 변환 및 출력
// 6. Table 출력
/////////////////////////////
int main()
{
	HuffmanEncoder encoder;
	encoder.readFile("input_data.txt");
	encoder.encoding();
	encoder.printHuffmanTable();

	return 0;
}