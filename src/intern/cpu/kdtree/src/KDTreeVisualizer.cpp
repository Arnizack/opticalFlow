#include "KDTreeVisualizer.h"
#include<iostream>
#include<fstream>

kdtree::KDTreeVisualizer::KDTreeVisualizer(KDTreeData& data)
{
	analyseTree(0,data);
}

void kdtree::KDTreeVisualizer::write(std::string filepath)
{
	std::ofstream file;
	file.open(filepath, std::ofstream::out);
	file << "graph G {" << std::endl;
	for (const auto& pair : NodePairs)
	{
		file << "\t" << pair.first << " -- " << pair.second << ";" << std::endl;
	}
	file << "}";
	file.close();
}

void kdtree::KDTreeVisualizer::analyseTree(uint32_t nodeIdx, KDTreeData& data, bool isLeaf)
{
	if (isLeaf)
		return;

	KdNode& node = data.kdTreeNodes[nodeIdx];

	std::string signiturRoot = std::to_string(nodeIdx);
	std::string signiturLeft;
	std::string signiturRight;

	signiturLeft = std::to_string(node.Left);
	signiturRight = std::to_string(node.Rigth);
	/*
	if (!node.IsLeftLeaf)
	{
		KdNode& leftNode = data.kdTreeNodes[node.Left];
		signiturLeft = std::to_string(node.Left);//nodeSignitur(leftNode);
	}
	else
		

	if (!node.IsRigthLeaf)
	{
		KdNode& rightNode = data.kdTreeNodes[node.Rigth];
		
	}
	else
		signiturRight = leafSignitur(data.samples[node.Rigth]);
	*/

	NodePairs.push_back(std::pair<std::string, std::string>(signiturRoot, signiturLeft));

	NodePairs.push_back(std::pair<std::string, std::string>(signiturRoot, signiturRight));

	analyseTree(node.Left, data, node.IsLeftLeaf);
	analyseTree(node.Rigth, data, node.IsRigthLeaf);
}

std::string kdtree::KDTreeVisualizer::nodeSignitur(KdNode& node)
{
	std::string dim = std::to_string((int)node.Dimension);
	std::string cut = std::to_string(node.Cut);
	return cut + "l" + dim;
}

std::string kdtree::KDTreeVisualizer::leafSignitur(KdTreeVal& leaf)
{
	std::string result = "";
	for (int i = 0; i < 5; i++)
	{
		result += std::to_string((int)leaf[i])+"l";
	}
	return result;
}
