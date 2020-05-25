#pragma once
#include"KDTreeData.h"
#include"loggerHelper.hpp"

namespace kdtree
{
	class KDTreeAnalyser
	{
	public:
		void logTreeLevelCount(KDTreeData& data, int Loglevel = 10);
		void logLeftPath(KDTreeData& data, int Loglevel = 10,int startNodeIdx=0);

	private:

		void logCountPerLevel(int LogLevel);
		void countLevels(uint32_t nodeIdx, KDTreeData& data, bool isLeaf = false, int level = 0);



		std::vector<int> CountPerLevel;
	};

}
                