#include "ColorSpaceConverter.hpp"
#include<math.h>
namespace visualization
{
    void _HSVtoRGB(float* r, float* g, float* b, float h, float s, float v) {
        int i;
        float f, p, q, t;
        if (s == 0) { // achromatisch (Grau)
            *r = *g = *b = v;
            return;
        }
        h /= 60;           // sector 0 to 5
        i = floor(h);
        f = h - i;         // factorial part of h
        p = v * (1 - s);
        q = v * (1 - s * f);
        t = v * (1 - s * (1 - f));
        switch (i) {
        case 0: *r = v; *g = t; *b = p; break;
        case 1: *r = q; *g = v; *b = p; break;
        case 2: *r = p; *g = v; *b = t; break;
        case 3: *r = p; *g = q; *b = v; break;
        case 4: *r = t; *g = p; *b = v; break;
        default:  // case 5:
            *r = v; *g = p; *b = q; break;
        }
    }
    core::Color ColorSpaceConverter::HSVToRGB(const HSV& hsv) const
    {
        float r, g, b;
        _HSVtoRGB(&r, &g, &b, hsv.H, hsv.S, hsv.V);
        r *= 255;
        g *= 255;
        b *= 255;
        return core::Color(r, g, b);
    }
}
