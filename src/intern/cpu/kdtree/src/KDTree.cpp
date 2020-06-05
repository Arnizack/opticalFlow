#include"KDTree.h"
#include"KdNode.h"
#include"MortonNode.hpp"
#include"CoordinateConverter.hpp"
#include<numeric>
#include<intrin.h>
#include<math.h>
#include"KDTreeFactory.hpp"
#include"IKDTreeQuery.h"
#include"KDTreeQuery/KDTreeRecrusivQuery.hpp"
#include <iostream>
#include"logger.hpp"
#include"KDTreeQuery/KDTreeSerialQuery.hpp"


namespace kdtree
{
	

	kdtree::KDResult::KDResult(uint32_t x, uint32_t y, unsigned char r, unsigned char g, unsigned char b, float weight, int level)
	{
		X = x;
		Y = y;
		R = r;
		G = g;
		B = b;
		Weight = weight;
		Level = level;
	}

	KDResult::KDResult()
	{
		X = 0;
		Y = 0;
		R = 0;
		G = 0;
		B = 0;
		Weight = 0;
	}

	

	
	
	std::vector<KDResult> queryKDTree(KDTreeData& data, uint32_t x, uint32_t y, core::Color color)
	{
		std::vector<KDResult> results;
		results.reserve(32);
		float sigma =static_cast<float>(3 / sqrt(11));

		data.Query->Query(StandardVal(x,y,color.Red,color.Green,color.Blue),data.kdTreeNodes,data.samples,results, sigma,data.sampleCount);


		return results;
	}

	KDTree::KDTree(std::shared_ptr<core::ImageRGB> image)
	{
		Image = image;
	}


	KDTreeData KDTree::Build(float sigma_distance, float sigma_color, int sampleCount)
	{
		uint32_t width = Image->GetWidth();
		uint32_t height = Image->GetHeight();
		uint32_t nodeCount = width * height;
		StandardVal maxValues(width, height, (unsigned char)255, (unsigned char)255, (unsigned char)255);
		CoordinateConverter converter(maxValues, sigma_distance, sigma_color);

		std::unique_ptr<IKDTreeQuery> queryPtr = std::make_unique< KDTreeRecrusivQuery>(width, height, sampleCount, converter);
		//std::unique_ptr<IKDTreeQuery> queryPtr = std::make_unique< KDTreeSerialQuery>(width, height, sampleCount, converter);

		KDTreeData data(std::move(queryPtr));
		
		data.sampleCount = sampleCount;

		std::vector<MortonNode>  mortonNodes(nodeCount-1);
		std::vector<MortonCodeVal> mortonCodes(nodeCount);
		auto samples = std::vector<KDTreeSample>(nodeCount);

		data.kdTreeNodes = std::vector<KdNode>(nodeCount-1);
		auto& kdTreeNodes = data.kdTreeNodes;

		KDTreeFactory factory = KDTreeFactory();
		

		//fill vector
		for (uint32_t x = 0; x < width;x++)
		{
			for (uint32_t y = 0; y < height; y++)
			{
				uint32_t listIdx = x* height + y;
				//logger::log(10, "%d; %d; %d",x,y, listIdx);
				core::Color color = Image->GetPixel(x, y);
				StandardVal pixel(x, y, color.Red, color.Green, color.Blue);
				samples[listIdx] = converter.StandardToKdTree(pixel);
				mortonCodes[listIdx] = converter.StandardToMortonCode(pixel);
				samples[listIdx].mortonCode = mortonCodes[listIdx].Value;
			}

		}
		//sort
		factory.sortMortonCodeAndSamples(mortonCodes, samples);
		//remove Duplications
		factory.makeMortonCodesUnique(mortonCodes);

		factory.buildMortonNodes(mortonCodes, mortonNodes, samples);

		factory.computeBoundingBoxes(mortonNodes, samples);

		factory.createKdTreeNodes(mortonNodes, mortonCodes, samples, kdTreeNodes, converter);

		data.samples = std::vector<KdTreeVal>(samples.size());

		for (uint32_t i = 0; i < samples.size(); i++)
		{
			data.samples[i] = samples[i];
		}

		return data;
	}
	
}