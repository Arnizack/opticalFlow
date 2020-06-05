#include"KDTreeFactory.hpp"
#include<numeric>
#include<intrin.h>
#include<math.h>
#include <stdexcept>
namespace kdtree
{

	KDTreeSample::KDTreeSample(KdTreeVal kdVal)
	{
		X = kdVal.X;
		Y = kdVal.Y;
		R = kdVal.R;
		G = kdVal.G;
		B = kdVal.B;
	}
	KDTreeSample::KDTreeSample()
	{
		X = 0;
		Y = 0;
		R = 0;
		G = 0;
		B = 0;
	}

	template <typename T, typename Compare>
	std::vector<std::size_t> sort_permutation(
		const std::vector<T>& vec,
		Compare& compare)
	{
		std::vector<std::size_t> p(vec.size());
		std::iota(p.begin(), p.end(), 0);
		std::sort(p.begin(), p.end(),
			[&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
		return p;
	}

	template <typename T>
	std::vector<T> apply_permutation(
		const std::vector<T>& vec,
		const std::vector<std::size_t>& p)
	{
		std::vector<T> sorted_vec(vec.size());
		std::transform(p.begin(), p.end(), sorted_vec.begin(),
			[&](std::size_t i) { return vec[i]; });
		return sorted_vec;
	}


	void KDTreeFactory::sortMortonCodeAndSamples(std::vector<MortonCodeVal>& mortonCodes, std::vector<KDTreeSample>& samples)
	{
		auto permulation = sort_permutation(mortonCodes, [&](MortonCodeVal i, MortonCodeVal j) {return i.Value < j.Value; });
		mortonCodes = apply_permutation(mortonCodes, permulation);
		samples = apply_permutation(samples, permulation);
		
	}

	void KDTreeFactory::makeMortonCodesUnique(std::vector<MortonCodeVal>& sortedCodes)
	{
		std::vector<MortonCodeVal> sortedCodesResult = sortedCodes;
		for (int i = 0; i < sortedCodes.size(); i++)
		{
			uint8_t numberOfSameValues = 0;
			for (int j = i - 1; j >= 0 && sortedCodes[i].Value == sortedCodes[j].Value; j--)
			{
				//uint8_t overflow
				if ((1 << 8) - 1 == numberOfSameValues)
				{
					throw std::overflow_error("to many equal MortonCode Values");
				}
				numberOfSameValues++;
			}
			uint16_t label = numberOfSameValues;
			label <<= 6;
			sortedCodesResult[i].Value += label;

		}
		sortedCodes = sortedCodesResult;
	}

	int8_t KDTreeFactory::findCommanPrefix(std::vector<MortonCodeVal>& mortonCodes, uint32_t a, uint32_t b)
	{
		//Condition: a is in the range of the mortonCodes 
		if (0 > b || b >= mortonCodes.size())
			return -1;
		uint64_t xorOp = mortonCodes[a].Value ^ mortonCodes[b].Value;
		int64_t result =  __lzcnt64(xorOp);
		return static_cast<uint8_t>(result);
	}

	short KDTreeFactory::determineDirection(std::vector<MortonCodeVal>& mortonCodes, uint32_t index)
	{
		auto commandPrefixLeft = findCommanPrefix(mortonCodes, index, index - 1);
		auto commandPrefixRight = findCommanPrefix(mortonCodes, index, index + 1);

		if (commandPrefixRight > commandPrefixLeft)
		{
			return 1;
		}
		else
		{
			return -1;
		}

	}

	std::array<uint32_t, 2> KDTreeFactory::determineRange(std::vector<MortonCodeVal>& mortonCodes, uint32_t i)
	{
		
		if (i == 0)
			return { 0,static_cast<uint32_t>(mortonCodes.size()-1) };
		
		//determine direction
		short direction = determineDirection(mortonCodes, i);

		//Compute upper bound for the length of the range
		uint8_t commanPrefix_min = findCommanPrefix(mortonCodes, i, i - direction);

		uint32_t length_max = 2;

		while (findCommanPrefix(mortonCodes, i, i + length_max * direction) > commanPrefix_min)
			length_max *= 2;

		//find the other end using binary search
		/*
		Example:
		Input list: 0,3,5,6,7,18,29,48,49
		Search Element 29
		l_max = 9
		l = 0
		for t <- {l_max / 2, l_max/4, ..., 1}
				= {5,3,1}

		t = 5, l = 0
		0,3,5,6,7,18,29,48,49
				  ^
				  |
		t = 3, l = 5
		0,3,5,6,7,18,29,48,49
						 ^
						 |
		t = 1, l = 5
		0,3,5,6,7,18,29,48,49
					  ^
					  |
		found 29

		*/
		uint32_t length = 0;
		for (uint32_t t = (float)length_max / 2.0; t * 2 > 1; t /= 2)
		{
			if (findCommanPrefix(mortonCodes, i, i + (length + static_cast<uint32_t>(ceil(t)))*direction) > commanPrefix_min)
			{
				length += static_cast<uint32_t>(ceil(t));
				
			}
		}

		uint32_t otherEnd = i + length * direction;

		if (otherEnd < i)
			return { otherEnd,i };
		return{ i,otherEnd };

	}

	uint32_t KDTreeFactory::findSplit(std::vector<MortonCodeVal>& mortonCodes, uint32_t first, uint32_t last)
	{
		uint8_t commanPrefix_min = findCommanPrefix(mortonCodes, first, last);
		uint32_t split = 0;
		float length = last - first;
		for (float t = length / 2; t * 2 > 1; t /= 2)
		{
			int8_t prefix = findCommanPrefix(mortonCodes, first, first + split + ceil(t));
			if ( prefix > commanPrefix_min)
				split += ceil(t);
		}
		uint32_t splitPosition = split + first;
		return splitPosition;
	}

	void KDTreeFactory::buildMortonNodes(std::vector<MortonCodeVal>& mortonCodes, std::vector<MortonNode>& mortonNodes, std::vector<KDTreeSample>& samples)
	{
		for (int i = 0; i < mortonCodes.size()-1; i++) // in parrallel in Cuda
		{
			std::array<uint32_t, 2> range = determineRange(mortonCodes, i);
			uint32_t first = range[0];
			uint32_t last = range[1];

			uint32_t split = findSplit(mortonCodes, first, last);

			mortonNodes[i].First = first;
			mortonNodes[i].Last = last;
			mortonNodes[i].Split = split;


			//if child is leave
			if (split + 1 == last)
				samples[last].Parent = &mortonNodes[i];
			else
				mortonNodes[split + 1].Parent = &mortonNodes[i];


			if (split == first)
				samples[first].Parent = &mortonNodes[i];
			else
				mortonNodes[split].Parent = &mortonNodes[i];



		}
	}

	void KDTreeFactory::computeBoundingBoxes(std::vector<MortonNode>& mortonNodes, std::vector<KDTreeSample>& samples)
	{
		for (KDTreeSample& sample : samples) // in parrallel in Cuda
		{
			MortonNode* node = sample.Parent;
			BoundingBox* bound = &BoundingBox();
			for (int i = 0; i < 5; i++)
			{
				float val = sample.operator[](i);
				bound->Max[i] = val;
				bound->Min[i] = val;
			}
			bool shouldRun = false;
			do
			{
				if (node->AlreadyWorkedOn)
					node->Bounds.enlarge(*bound);
				else
					node->Bounds = BoundingBox(*bound);
				bound = &node->Bounds;
				//to ensure, that it only moves to the Parent, when the full BoundingBox was computed
				shouldRun = node->AlreadyWorkedOn;
				node->AlreadyWorkedOn = true;
				if (node->Parent != nullptr)
					node = node->Parent;
				else
					shouldRun = false;

			} while (shouldRun);
		}
	}

	float KDTreeFactory::findCut(CoordinateConverter& converter, uint64_t mortonCode, uint8_t prefix, short dimension)
	{

		uint64_t mask = 0xffffffffffffffffu;
		if (prefix == 0)
			mask = 0;
		else
			mask <<= 64 - prefix;
		mortonCode &= mask;
		uint8_t move = 64 - prefix - 1;
		uint64_t one = static_cast<uint64_t>(1) << move;
		mortonCode |= one;
		KdTreeVal cutPoint = converter.MortonCodeToKdTree(mortonCode);
		return cutPoint[dimension];
	}

	void KDTreeFactory::createKdTreeNodes(std::vector<MortonNode>& mortonNodes, std::vector<MortonCodeVal>& mortonCodes, std::vector<KDTreeSample>& samples, std::vector<KdNode>& kdTreeNodes,
		CoordinateConverter& converter)
	{
		for (int i = 0; i < mortonNodes.size(); i++)
		{
			KdNode& kdTreeNode = kdTreeNodes[i];
			MortonNode& mortonNode = mortonNodes[i];

			uint32_t split = mortonNode.Split;
			uint8_t prefix = findCommanPrefix(mortonCodes, split, split + 1);
			kdTreeNode.Dimension = prefix % 5;

			uint64_t splitValue = mortonCodes[split].Value;

			kdTreeNode.Cut = findCut(converter, splitValue, prefix, kdTreeNode.Dimension);
			kdTreeNode.Max = mortonNode.Bounds.Max[kdTreeNode.Dimension];
			kdTreeNode.Min = mortonNode.Bounds.Min[kdTreeNode.Dimension];

			if (split + 1 == mortonNode.Last)
				kdTreeNode.IsRigthLeaf = true;
			if (split == mortonNode.First)
				kdTreeNode.IsLeftLeaf = true;

			kdTreeNode.Left = mortonNode.Split;
			kdTreeNode.Rigth = mortonNode.Split + 1;

			if (kdTreeNode.IsLeftLeaf)
				kdTreeNode.leftLeaf = &samples[kdTreeNode.Left];
			else
				kdTreeNode.leftPtr = &kdTreeNodes[kdTreeNode.Left];

			if (kdTreeNode.IsRigthLeaf)
				kdTreeNode.rightLeaf = &samples[kdTreeNode.Rigth];
			else
				kdTreeNode.rightPtr = &kdTreeNodes[kdTreeNode.Rigth];


		}
	}


}