# TOGMatcher

This is a project for doing experiments with OpenCV template matching using X and Y gradient images as the templates.  It also has code for doing experiments with template matching for finding colored landmarks.  The landmarks are 2x2 grids where each block has a color with RGB components that are either 255 or 0.  The corresponding colors are black, red, green, yellow, blue, magenta, cyan, and white.

# Installation

The project compiles in the Community edition of Visual Studio 2019 (VS 2019).  It uses the Windows pre-built OpenCV 4.1.0 libraries extracted to **c:\opencv-4.1.0**.  I just copied the appropriate OpenCV DLLs to wherever I had my executables.  The project creates a command-line Windows executable.  I have tested it on a Windows 10 64-bit machine.

I originally developed the code in Visual Studio 2015 on a Windows 7 64-bit machine with Service Pack 1.  I left the old project files in the repo for posterity.  All new work will use the VS 2019 project files.

## Camera

I tested with a Logitech c270.  It was the cheapest one I could find that I could purchase locally.  It was plug-and-play.
