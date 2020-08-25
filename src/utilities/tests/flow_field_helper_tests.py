from src.utilities.flow_field_helper import *

class FlowFieldHelperTest:
    def read_flow_field(self):

        flow_field = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\Dimetrodon\flow10.flo")
        print(flow_field.shape, (2,388,584))
        width = flow_field.shape[1]
        height = flow_field.shape[2]
        show_flow_field(flow_field,width,height)
        flow_field2 = read_flow_field(
            r"..\..\..\resources\eval-twoframes-groundtruth\Grove2\flow10.flo")
        show_flow_difference(flow_field,flow_field2, width, height)


if __name__ == '__main__':
    FlowFieldHelperTest().read_flow_field()
