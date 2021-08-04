#include <string>
#include <OpenImageIO/imageio.h>

using namespace std;
using namespace OIIO;

int  main(int argc, char const *argv[])
{
    string filename = "/home/chenyu/图片/colmap.png";
    auto in = ImageInput::open(filename);
    
    if (!in) return 0;

    const ImageSpec &spec = in->spec();
    int xres = spec.width;
    int yres = spec.height;
    int channels = spec.nchannels;
    std::vector<unsigned char> pixels(xres * yres * channels);
    in->read_image(TypeDesc::UINT8, &pixels[0]);
    in->close();
}
