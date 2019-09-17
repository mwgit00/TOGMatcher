# TOGMatcher

This is a project for doing experiments with OpenCV template matching using X and Y gradient images as the templates.  It also has code for doing experiments with template matching for finding colored landmarks.  The landmarks are 2x2 grids with two black squares at the diagonals as in a classic checkerboard.  The other squares are colored yellow, magenta, or cyan.  These are bright colors with one B,G,R component set to 0.  This provides good contrast with the black squares and limiting the number of colors makes it easier to identify them.  The colored squares must be different from one another.  This scheme only produces 12 unique patterns but the detection performance seems to be reasonably good.  The code has routines for creating PNG files that can be printed for use as landmarks in a lab, home, etc.

# Installation

The project compiles in the Community edition of Visual Studio 2019 (VS 2019).  It uses the Windows pre-built OpenCV 4.1.0 libraries extracted to **c:\opencv-4.1.0**.  I just copied the appropriate OpenCV DLLs to wherever I had my executables.  The project creates a command-line Windows executable.  I have tested it on a Windows 10 64-bit machine.

I originally developed the code in Visual Studio 2015 on a Windows 7 64-bit machine with Service Pack 1.  I left the old project files in the repo for posterity.  New work will mostly use the VS 2019 project files.  I may update the VS 2015 project if I feel so inclined.

## Camera

I tested with a Logitech c270.  It was the cheapest one I could find that I could purchase locally.  It was plug-and-play.
