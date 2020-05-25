#include "KDTreeAnalyser.hpp"
#include"KdNode.h"


namespace kdtree
{
	void KDTreeAnalyser::logTreeLevelCount(KDTreeData& data, int Loglevel)
	{
		countLevels(0,data);
		logCountPerLevel(Loglevel);
	}
	void KDTreeAnalyser::logLeftPath(KDTreeData& data, int Loglevel, int startNodeIdx)
	{
		KdNode& node = data.kdTreeNodes[startNodeIdx];
		logger::log(Loglevel, "Dimension: %d Cut: %f Max: %f Min: %f", node.Dimension, node.Cut,node.Min,node.Max);
		if (node.IsLeftLeaf)
			logger::log(Loglevel, "Leaf %f, %f, %f,%f,%f"
				, data.samples[node.Left][0], data.samples[node.Left][1], data.samples[node.Left][2], data.samples[node.Left][3], data.samples[node.Left][4]);
		else
		{
			logLeftPath(data, Loglevel, node.Left);
		}
	}

	

	void KDTreeAnalyser::logCountPerLevel(int LogLevel)
	{
		for (int count : CountPerLevel)
		{
			logger::log(LogLevel, "%d", count);
		}
	}
	void KDTreeAnalyser::countLevels(uint32_t nodeIdx, KDTreeData& data, bool isLeaf, int level)
	{
		

		while (level >= CountPerLevel.size())
		{
			CountPerLevel.push_back(0);
		}

		CountPerLevel[level]++;

		if (isLeaf)
			return;
		
		KdNode& node = data.kdTreeNodes[nodeIdx];

		countLevels(node.Left, data, node.IsLeftLeaf, level + 1);
		countLevels(node.Rigth, data, node.IsRigthLeaf, level + 1);



	}
}
