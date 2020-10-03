
#include"core/IArray.h"
#include"gmock/gmock.h"
#include"MockIArray.h"



TEST(TestCaseName, TestName) {
	
	core::testing::MockIArray<int, 3> mock_array;

	core::IArray<int, 3>* test_array = & mock_array;

	EXPECT_CALL(mock_array, Size());
	int test = test_array->Size();

	

}