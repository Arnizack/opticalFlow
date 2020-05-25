#include"KDTreeData.h"
namespace kdtree
{
	KDTreeData::KDTreeData(std::unique_ptr<IKDTreeQuery> query)
	{
		Query = std::move(query);
	}
}
