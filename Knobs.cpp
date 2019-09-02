// MIT License
//
// Copyright(c) 2018 Mark Whitney
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <string>
#include "Knobs.h"


Knobs::Knobs() :
    is_op_required(false),
    is_equ_hist_enabled(false),
    is_mask_enabled(false),
    is_record_enabled(false),
    kpreblur(1),
    kcliplimit(4),
    nchannel(Knobs::ALL_CHANNELS),
    noutmode(Knobs::OUT_COLOR),
    op_id(Knobs::OP_NONE),
    nimgscale(6),
    nksize(1),
    vimgscale({ 0.25, 0.325, 0.4, 0.5, 0.625, 0.75, 1.0 }),
    vksize({ -1, 1, 3, 5, 7})
{
}


Knobs::~Knobs()
{
}


void Knobs::show_help(void) const
{
    std::cout << std::endl;
    std::cout << "KEY FUNCTION" << std::endl;
    std::cout << "--- ------------------------------------------------------" << std::endl;
    std::cout << "Esc Quit" << std::endl;
    std::cout << "1   Use Blue channel" << std::endl;
    std::cout << "2   Use Green channel" << std::endl;
    std::cout << "3   Use Red channel" << std::endl;
    std::cout << "4   Use all channels in grayscale image" << std::endl;
    std::cout << "8   Output raw template match result " << std::endl;
    std::cout << "9   Output masked match result on pre-processed gray image" << std::endl;
    std::cout << "0   Output best match result on color image" << std::endl;
    std::cout << "-   Decrease pre-blur" << std::endl;
    std::cout << "=   Increase pre-blur" << std::endl;
    std::cout << "_   Decrease CLAHE clip limit" << std::endl;
    std::cout << "+   Increase CLAHE clip limit" << std::endl;
    std::cout << "[   Decrease image scale" << std::endl;
    std::cout << "]   Increase image scale" << std::endl;
    std::cout << "{   Decrease Sobel kernel size" << std::endl;
    std::cout << "}   Increase Sobel kernel size" << std::endl;
    std::cout << "e   Toggle histogram equalization" << std::endl;
    std::cout << "m   Toggle mask mode for template matching" << std::endl;
    std::cout << "r   Toggle recording mode" << std::endl;
    std::cout << "t   Select next template from collection" << std::endl;
    std::cout << "v   Create video from files in movie folder" << std::endl;
    std::cout << "?   Display this help info" << std::endl;
    std::cout << std::endl;
}


void Knobs::handle_keypress(const char ckey)
{
    bool is_valid = true;
    
    is_op_required = false;
    
    switch (ckey)
    {
        case '1':
        case '2':
        case '3':
        case '4':
        {
            // convert to channel code 0,1,2,3
            set_channel(ckey - '1');
            break;
        }
        case '7':
        {
            set_output_mode(Knobs::OUT_AUX);
            break;
        }
        case '8':
        {
            set_output_mode(Knobs::OUT_RAW);
            break;
        }
        case '9':
        {
            set_output_mode(Knobs::OUT_MASK);
            break;
        }
        case '0':
        {
            set_output_mode(Knobs::OUT_COLOR);
            break;
        }
        case '+':
        {
            inc_clip_limit();
            break;
        }
        case '_':
        {
            dec_clip_limit();
            break;
        }
        case '=':
        {
            inc_pre_blur();
            break;
        }
        case '-':
        {
            dec_pre_blur();
            break;
        }
        case ']':
        {
            inc_img_scale();
            break;
        }
        case '[':
        {
            dec_img_scale();
            break;
        }
        case '}':
        {
            inc_ksize();
            is_op_required = true;
            op_id = Knobs::OP_KSIZE;
            break;
        }
        case '{':
        {
            dec_ksize();
            is_op_required = true;
            op_id = Knobs::OP_KSIZE;
            break;
        }
        case 'e':
        {
            toggle_equ_hist_enabled();
            break;
        }
        case 'm':
        {
            toggle_mask_enabled();
            break;
        }
        case 'r':
        {
            is_op_required = true;
            op_id = Knobs::OP_RECORD;
            toggle_record_enabled();
            break;
        }
        case 't':
        {
            is_op_required = true;
            op_id = Knobs::OP_TEMPLATE;
            break;
        }
        case 'v':
        {
            is_op_required = true;
            op_id = Knobs::OP_MAKE_VIDEO;
            break;
        }
        case '?':
        {
            is_valid = false;
            show_help();
            break;
        }
        default:
        {
            is_valid = false;
            break;
        }
    }

    // display settings whenever valid keypress handled
    // except if it's an "op required" keypress
    if (is_valid && !is_op_required)
    {
        const std::vector<std::string> srgb({ "Blue ", "Green", "Red  ", "Gray " });
        const std::vector<std::string> sout({ "Raw  ", "Mask ", "Color", "Aux" });
        std::cout << "Equ=" << is_equ_hist_enabled;
        std::cout << "  Mask=" << is_mask_enabled;
        std::cout << "  Blur=" << kpreblur;
        std::cout << "  Clip=" << kcliplimit;
        std::cout << "  Ch=" << srgb[nchannel];
        std::cout << "  Out=" << sout[noutmode];
        std::cout << "  Scale=" << vimgscale[nimgscale];
        std::cout << std::endl;
    }
}


bool Knobs::get_op_flag(int& ropid)
{
    bool result = is_op_required;
    ropid = op_id;
    is_op_required = false;
    return result;
}