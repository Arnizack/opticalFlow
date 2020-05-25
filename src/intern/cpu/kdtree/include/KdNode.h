#pragma once
#include<memory>
#include<array>
#include"KDTreeSpaces.h"

class KdNode
{
public:
	uint32_t Left;
	uint32_t Rigth;
	bool IsLeftLeaf = false;
	bool IsRigthLeaf = false;

	short Dimension;

	float Cut;
	float Min;
	float Max;

	KdNode* leftPtr;
	KdNode* rightPtr;
	kdtree::KdTreeVal* leftLeaf;
	kdtree::KdTreeVal* rightLeaf;
	

};