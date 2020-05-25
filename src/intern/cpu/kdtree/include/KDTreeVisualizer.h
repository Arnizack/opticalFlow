#include"KDTreeData.h"
#include<string>

namespace kdtree
{
	class KDTreeVisualizer
	{
	public:
		KDTreeVisualizer(KDTreeData& data);
		void write(std::string filepath);
	private:
		void analyseTree(uint32_t nodeIdx, KDTreeData& data, bool isLeaf = false);
		
		std::string nodeSignitur(KdNode& node);
		std::string leafSignitur(KdTreeVal& leaf);

		std::vector<std::pair<std::string, std::string>> NodePairs;
	};
}