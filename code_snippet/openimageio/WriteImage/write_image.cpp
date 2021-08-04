#include <OpenImageIO/imageio.h>
#include <memory>
using namespace OIIO;

int main()
{
    const char *filename = "foo.jpg";
    const int xres = 640, yres = 480;
    const int channels = 3;
    unsigned char pixels[xres * yres * channels];

    ImageOutput* out = ImageOutput::create(filename);
    if (!out) return 0;
    ImageSpec spec(xres, yres, channels, TypeDesc::UINT8);
    out->open(filename, spec);
    out->write_image(TypeDesc::UINT8, pixels);
    out->close();
    delete out;
}