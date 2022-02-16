#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
//#include <cstring>
using namespace std;

class Node
{
private:
	int bit;
	char ascii;
	Node* leftNode;
	Node* rightNode;
public:
	Node(int bit)
		:bit(bit), ascii(NULL), leftNode(NULL), rightNode(NULL) {}
	Node(int bit, char ascii)
		:bit(bit), ascii(ascii), leftNode(NULL), rightNode(NULL) {}
	~Node() { delete leftNode; delete rightNode; }

	void setASCII(char ch) { ascii = ch; }
	void setLeft(Node* left) { leftNode = left; }
	void setRight(Node* right) { rightNode = right; }

	double getBit() { return bit; }
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

class HuffmanDecoder
{
private:
	string codeFile, tableFile;
	pair<int, unsigned short> huffmanTable[129];
	Node* huffmanTree;
	char* text;
	int fileSize;
	unsigned char *codeData;
public:
	HuffmanDecoder() :codeFile("huffman_code.hbs"), tableFile("huffman_table.hbs"), text(NULL), huffmanTable(), huffmanTree(NULL), fileSize(0), codeData(NULL) { }
	~HuffmanDecoder() { delete[] codeData; }

	// read code file
	bool readHuffmanCode();

	// read table file
	bool readHuffmanTable();

	// make table from table file
	void getHuffmanTable(unsigned char*, int);

	// make huffman tree from table
	void getHuffmanTree();

	// make text of input code
	bool getText();

	// store text
	bool storeText();
	
	// Do Decoding
	void decoding();

	// print huffman table
	void printHuffmanTable();
};

bool HuffmanDecoder::readHuffmanCode()
{
	ifstream read;
	read.open(codeFile);

	// file is not opened
	if (read.fail())
	{
		return false;
	}
	
	read.seekg(0, read.end);
	fileSize = (int)read.tellg();
	read.seekg(0, read.beg);

	codeData = new unsigned char[fileSize];
	read.read((char*)codeData, fileSize);
	read.close();
	return true;
}

bool HuffmanDecoder::readHuffmanTable()
{
	ifstream read;
	read.open(tableFile);

	// file is not opened
	if (read.fail())
	{
		return false;
	}

	read.seekg(0, read.end);
	int tableSize = (int)read.tellg();
	read.seekg(0, read.beg);

	unsigned char* tableData = new unsigned char[tableSize];
	read.read((char*)tableData, tableSize);
	read.close();
	
	getHuffmanTable(tableData, tableSize);
	delete[] tableData;
	getHuffmanTree();
	return true;
}

void HuffmanDecoder::getHuffmanTable(unsigned char tableData[], int dataSize)
{
	int usedBit = 0;
	int i = 0;
	while (i < dataSize)
	{
		// init all
		char ch = 0;
		int codeSize = 0;
		unsigned short code = 0;
		usedBit %= 8;

		// char
		ch |= tableData[i] << usedBit;
		ch |= tableData[i + 1] >> (8 - usedBit);

		// huffman code SIze
		codeSize |= (unsigned char)(tableData[i + 1] << usedBit);
		codeSize |= tableData[i + 2] >> (8 - usedBit);

		// codeSize < can use bit
		if (codeSize < 8 - usedBit)
		{
			code |= (unsigned char)((tableData[i + 2] << usedBit) >> (8 - codeSize));
			i = i + 2;
		}
		// codeSize < can use bit + 1 byte
		else if (codeSize < 16 - usedBit)
		{
			code |= (unsigned char)((tableData[i + 2] << usedBit) >> usedBit);
			// code = code << will use bit | tableData's unuse bit = 0
			code = (code << (codeSize + usedBit - 8)) | (unsigned char)(tableData[i + 3] >> (16 - codeSize - usedBit));
			i = i + 3;
		}
		// codeSize > can use bit + 1 byte
		else
		{
			code |= (unsigned char)((tableData[i + 2] << usedBit) >> usedBit);
			// code = code << 8 | tableData
			code = (code << 8) | (unsigned char)tableData[i + 3];
			// code = code << will use bit | tableData's unuse bit = 0
			code = (code << (codeSize + usedBit - 16)) | (unsigned char)(tableData[i + 4] >> (codeSize + usedBit - 16));
			i = i + 4;
		}
		usedBit += codeSize;

		if (ch == NULL)	// EOD
			huffmanTable[128] = pair<int, unsigned short>(codeSize, code);
		else			// ASCII
			huffmanTable[ch] = pair<int, unsigned short>(codeSize, code);
	}
}

void HuffmanDecoder::getHuffmanTree()
{
	// create root
	if (!huffmanTree)
		huffmanTree = new Node(-1);

	// huffman table loop
	for (int i = 0; i < 129; i++)
	{
		Node* node = huffmanTree;
		unsigned short code = huffmanTable[i].second;
		unsigned short checkBit = 1 << (huffmanTable[i].first - 1);
		while (checkBit != 0)
		{
			if (checkBit & code)
			{ // 1
				if (node->getRight() == NULL)
				{
					node->setRight(new Node(1));
				}
				node = node->getRight();
			}
			else
			{ // 0
				if (node->getLeft() == NULL)
				{
					node->setLeft(new Node(0));
				}
				node = node->getLeft();
			}
			checkBit = checkBit >> 1;
		}
		if ((char)i == NULL)
			node->setASCII(NULL);
		else
			node->setASCII((char)i);
	}
}

bool HuffmanDecoder::getText()
{
	text = new char[fileSize * 2];
	for (int i = 0; i < fileSize; i++)
	{
		text[i] = NULL;
	}
	int point = 0;

	Node* node = huffmanTree;
	for (int i = 0; i < fileSize; i++)
	{
		unsigned char readBit = 1 << 7;

		// read bit is not 0
		while (readBit)
		{
			if (readBit & codeData[i])
			{ // 1, go right
				node = node->getRight();
			}
			else
			{ // 0, go left
				node = node->getLeft();
			}

			// error
			if (node == NULL)
				return false;

			// no children
			if (!node->hasLeft() && !node->hasRight())
			{
				if (node->getASCII() == NULL)
				{ // EOD
					return true;
				}
				
				// text++
				text[point++] = node->getASCII();
				
				// tree restart
				node = huffmanTree;
			}
			readBit = readBit >> 1;
		} // readBit while
	} // fileSize for
	return false;
}

bool HuffmanDecoder::storeText()
{
	ofstream write;
	write.open("output.txt");

	// write
	write << text;
	write.close();

	return true;
}

void HuffmanDecoder::decoding()
{
	if (!readHuffmanCode()) {
		cout << "readHuffmanCode: Error" << endl;
		return;
	}
	if (!readHuffmanTable()) {
		cout << "readHuffmanTable: Error" << endl;
		return;
	}
	if (!getText()) {
		cout << "getText: Error" << endl;
		//return;
	}
	if (!storeText()) {
		cout << "storeText: Error" << endl;
		return;
	}
}

void HuffmanDecoder::printHuffmanTable()
{
	for (int i = 0; i < 128; i++)
	{
		unsigned short p = 1;
		if (huffmanTable[i].first != 0) {
			printf("%x", i);
			cout << '\t' << huffmanTable[i].first << '\t' << (int)huffmanTable[i].second << '\t';
			p = p << (huffmanTable[i].first - 1);
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

int main()
{
	HuffmanDecoder decoder;
	decoder.decoding();
	decoder.printHuffmanTable();
	return 0;
}