#include<gtest/gtest.h>
#include"KDTreeFactory.hpp"

using namespace kdtree;

TEST(KDTree,FindCommanPrefixTestOutOfRange )
{
	int expected = -1;
	std::vector<MortonCodeVal> mortonCodes = { 12 };
	kdtree::KDTreeFactory factory;

	
	int actual = factory.findCommanPrefix(mortonCodes, 0, 1);
	EXPECT_EQ(expected, actual);
	
}

TEST(KDTree, FindCommanPrefixTestZero)
{
	int expected = 64;
	std::vector<MortonCodeVal> mortonCodes = { 0,0 };
	kdtree::KDTreeFactory factory;
	int actual = factory.findCommanPrefix(mortonCodes, 0, 1);
	EXPECT_EQ(expected, actual);
}

TEST(KDTree, FindCommanPrefixTest1)
{
	int expected = 48;
	/*
	a = 12223304 = '101110101000001101001000'
	b = 12190536 = '101110100000001101001000'

	a = '000 000 000 000 000 000 000 000 000 000 000 000 000 010 111 010 100 000 110 100 100 0'
	b = '000 000 000 000 000 000 000 000 000 000 000 000 000 010 111 010 000 000 110 100 100 0'
	                                                                     ^
														                 |
																		 49
	*/

	std::vector<MortonCodeVal> mortonCodes = { 12223304,12190536 };
	kdtree::KDTreeFactory factory;
	int actual = factory.findCommanPrefix(mortonCodes, 0, 1);
	EXPECT_EQ(expected, actual);
}

TEST(KDTree, DetermineDirection)
{
	/*
	Morton Codes: 
	000 | 0
	001 | 1
	010 | 2

	*/
	std::vector<MortonCodeVal> mortonCodes = { 0,1,2 };
	kdtree::KDTreeFactory factory;
	std::vector<short> expecedDirections = { 1,-1 };
	for (int i = 0; i < expecedDirections.size(); i++)
	{
		short expeced = expecedDirections[i];
		short actual = factory.determineDirection(mortonCodes, i);
		EXPECT_EQ(expeced, actual);
		
	}
	
}

TEST(KDTree, SortMortonCodes)
{
	using Sample = kdtree::KDTreeSample;
	std::vector<MortonCodeVal> mortonCodes = { 4,67,3,4,6,2,4 };
	std::vector<Sample> samples;
	for (MortonCodeVal mortonCode : mortonCodes)
	{
		Sample sample;
		sample.X = static_cast<float>(mortonCode.Value);
		samples.push_back(sample);
	}
	kdtree::KDTreeFactory factory;
	factory.sortMortonCodeAndSamples(mortonCodes, samples);

	std::vector<MortonCodeVal> ExpectedMortonCodes = { 2,3,4,4,4,6,67 };

	for (int i = 0; i < ExpectedMortonCodes.size(); i++)
	{
		EXPECT_EQ(ExpectedMortonCodes[i].Value, mortonCodes[i].Value);
		EXPECT_NEAR((float)ExpectedMortonCodes[i].Value, samples[i].X,0.01);

	}
}

TEST(KDTree, MakeMortonCodeUnique)
{
	kdtree::KDTreeFactory factory;
	/*
	mortonCodes:
	000 001 000 000 000 = 512
	000 001 000 000 000 = 512
	000 001 000 000 000 = 512
	000 001 000 000 000 = 512
	000 001 000 000 000 = 512
	000 001 000 000 000 = 512
	000 001 000 000 000 = 512
	000 001 000 000 000 = 512
	000 010 000 000 000 = 1024
	000 010 000 000 000 = 1024
	000 010 000 000 000 = 1024
	000 011 000 000 000 = 1536
	000 011 000 000 000 = 1536
	000 111 000 000 000 = 3584
	000 111 000 000 000 = 3584

	Expected Result:

	000 001 000 000 000 = 000001000000000 = 512
	000 001 001 000 000 = 000001001000000 = 576
	000 001 010 000 000 = 000001010000000 = 640
	000 001 011 000 000 = 000001011000000 = 704
	000 001 100 000 000 = 000001100000000 = 768
	000 001 101 000 000 = 000001101000000 = 832
	000 001 110 000 000 = 000001110000000 = 896
	000 001 111 000 000 = 000001111000000 = 960
	000 010 000 000 000 = 000010000000000 = 1024
	000 010 001 000 000 = 000010001000000 = 1088
	000 010 010 000 000 = 000010010000000 = 1152

	000 011 000 000 000 = 000011000000000 = 1536
	000 011 001 000 000 = 000011001000000 = 1600

	000 111 000 000 000 = 000111000000000 = 3584
	000 111 001 000 000 = 000111001000000 = 3648

	*/

	std::vector<MortonCodeVal> mortonCodes = 
	{ 512,512,512,512,512,512,512,512,1024,1024,1024,1536,1536,3584,3584 };
	

	std::vector<MortonCodeVal> ExpectedMortonCodes =
	{ 512,576,640,704,768,832,896,960,1024,1088,1152,1536,1600,3584,3648 };

	factory.makeMortonCodesUnique(mortonCodes);

	for (int i = 0; i < mortonCodes.size(); i++)
	{
		EXPECT_EQ(mortonCodes[i].Value, ExpectedMortonCodes[i].Value);
	}

}

TEST(KDTree, DetermineRange)
{
	/*

	0. 1  = 00001 <-|  |  |*
	1. 2  = 00010   |  |<-|* 
	2. 4  = 00100   |  |<-|*
	3. 5  = 00101   |<-|  |*
	4. 19 = 10011   |<-|* 
	5. 24 = 11000   |  |<-|  |*
	6. 25 = 11001   |  |  |<-|*
	7. 30 = 11110   |  |  |*  
	*/
	std::vector<MortonCodeVal> mortonCodes = { 1,2,4,5,19,24,25,30 };
	std::vector<std::array<uint32_t, 2>> expectedRanges =
	{ {0,7},{0,1},{2,3},{0,3},{4,7},{5,7},{5,6} };

	kdtree::KDTreeFactory factory;

	for (int i = 0; i < 7; i++)
	{
		auto range = factory.determineRange(mortonCodes, i);
		EXPECT_EQ(expectedRanges[i][0],range[0]);
		EXPECT_EQ(expectedRanges[i][1],range[1]);
	}


}

TEST(KDTree, DetermineSplit)
{
	/*

	0. 1  = 00001 <-|  |  |*
	1. 2  = 00010   |  |<-|*
	2. 4  = 00100   |  |<-|*
	3. 5  = 00101   |<-|  |*
	4. 19 = 10011   |<-|*
	5. 24 = 11000   |  |<-|  |*
	6. 25 = 11001   |  |  |<-|*
	7. 30 = 11110   |  |  |*
	*/
	std::vector<MortonCodeVal> mortonCodes = { 1,2,4,5,19,24,25,30 };
	std::vector<uint32_t> expectedSplit = { 3,0,2,1,4,6,5 };
	//                            Node ID:   0 1 2 3 4 5 6
	kdtree::KDTreeFactory factory;
	for (int i = 0;  i < 7; i++)
	{
		auto range = factory.determineRange(mortonCodes, i);

		auto split = factory.findSplit(mortonCodes, range[0], range[1]);
		EXPECT_EQ(expectedSplit[i], split);
	}

}

TEST(KDTree, findCut)
{
	kdtree::KDTreeFactory factory;
	/*
	CutValue = 1101001 ... = 1101001000 = 
			= 840
	Prefix = 32
	Dimension = 2
	MortonCode = 00100 00100 00000 00100 00000 00000 00000 00000 00000 00000 00000 00000 0000
			   = 0x2100400000000000u
	*/
	uint64_t mortonCode =  0x2100400000000000u ;
	float expectedCutValue = 840;
	int dimension = 2;
	CoordinateConverter converter(StandardVal(1024, 1024,224, 224, 224),1,1);
	float actualCutValue = factory.findCut(converter,mortonCode,32, dimension);
	EXPECT_NEAR(expectedCutValue, actualCutValue, 0.01);
}

TEST(KDTree, findCutZero)
{
	kdtree::KDTreeFactory factory;
	/*
	CutValue = 100000.... = 2^64 = 0x8000000000000000u
	Prefix = 0
	Dimension = 0
	MortonCode = 00100 00100 00000 00100 00000 00000 00000 00000 00000 00000 00000 00000 0000
			   = 0x2100400000000000u
	*/
	uint64_t mortonCode = 0x2100400000000000u;
	float expectedCutValue = 512;
	int dimension = 0;
	int prefix = 0;
	CoordinateConverter converter(StandardVal(1024, 1024, 224, 224, 224), 1, 1);
	float actualCutValue = factory.findCut(converter, mortonCode, prefix, dimension);
	EXPECT_NEAR(expectedCutValue, actualCutValue, 0.01);
}


TEST(KDTree, BuildNodesBoundingBox)
{
	kdtree::KDTreeFactory factory;
	/*

	0. 1  = 00001 <-|  |  |*
	1. 2  = 00010   |  |<-|*
	2. 4  = 00100   |  |<-|*
	3. 5  = 00101   |<-|  |*
	4. 19 = 10011   |<-|*
	5. 24 = 11000   |  |<-|  |*
	6. 25 = 11001   |  |  |<-|*
	7. 30 = 11110   |  |  |*
	*/
	std::vector<MortonCodeVal> mortonCodes = { 1,2,4,5,19,24,25,30 };

	std::vector<kdtree::KDTreeSample> samples(mortonCodes.size());
	std::vector<MortonNode> mortonNodes(mortonCodes.size() - 1);

	for (int i = 0; i < mortonCodes.size(); i++)
	{
		samples[i].X = static_cast<float>(i);
	}

	factory.buildMortonNodes(mortonCodes, mortonNodes, samples);
	factory.computeBoundingBoxes(mortonNodes, samples);

	std::vector<std::array<uint32_t, 2>> expectedRanges =
	{ {0,7},{0,1},{2,3},{0,3},{4,7},{5,7},{5,6} };

	std::vector<uint32_t> expectedSplit = { 3,0,2,1,4,6,5 };

	std::vector< MortonNode> expectedMortonNodes(mortonCodes.size() - 1);
	for (int i = 0; i < 7; i++)
	{
		expectedMortonNodes[i].First = expectedRanges[i][0];
		expectedMortonNodes[i].Last = expectedRanges[i][1];
		expectedMortonNodes[i].Bounds.Max[0] = static_cast<float>(expectedRanges[i][1]);
		expectedMortonNodes[i].Bounds.Min[0] = static_cast<float>(expectedRanges[i][0]);
		expectedMortonNodes[i].Split = expectedSplit[i];
	}

	//link
	auto& node0 = expectedMortonNodes[0];
	auto& node1 = expectedMortonNodes[1];
	auto& node2 = expectedMortonNodes[2];
	auto& node3 = expectedMortonNodes[3];
	auto& node4 = expectedMortonNodes[4];
	auto& node5 = expectedMortonNodes[5];
	auto& node6 = expectedMortonNodes[6];

	node1.Parent = &node3;
	node2.Parent = &node3;
	node3.Parent = &node0;
	node4.Parent = &node0;
	node5.Parent = &node4;
	node6.Parent = &node5;

	for (int i = 0; i < 6; i++)
	{
		auto actualNode = mortonNodes[i];
		auto expectedNode = expectedMortonNodes[i];
		EXPECT_EQ(actualNode.First, expectedNode.First);
		EXPECT_EQ(actualNode.Last, expectedNode.Last);
		EXPECT_EQ(actualNode.Split, expectedNode.Split);
		EXPECT_EQ(actualNode.Bounds.Max[0], expectedNode.Bounds.Max[0]);
		EXPECT_EQ(actualNode.Bounds.Min[0], expectedNode.Bounds.Min[0]);
		if (actualNode.Parent != nullptr)
		{
			EXPECT_EQ(actualNode.Parent->First, expectedNode.Parent->First);
			EXPECT_EQ(actualNode.Parent->Last, expectedNode.Parent->Last);
		}
	}
}

TEST(KDTree, BuildKdNodes)
{

}