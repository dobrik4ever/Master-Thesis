class Conv_Calc:

    def __init__(self):
        pass

    def calc_conv(self, input, padding, kernel, stride, dilation):
        # o = output
        i = input
        p = padding
        k = kernel
        s = stride
        d = dilation
        o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        return o


    def calc_upconv(self, input, padding, output_padding, kernel, stride, dilation):
        i = input
        p = padding
        k = kernel
        s = stride
        d = dilation
        o = (i -1)*s - 2*p + k + output_padding 