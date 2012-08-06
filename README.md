Key Detection
=============

Training the model
------------------

Run the `train` tool, providing an MP3 directory and a LAB directory.

The MP3 directory must be a parent directory that has a number of subdirectories containing MP3 files. 

The LAB directory must have the exact same structure as the MP3 directory, with exactly the same file names, except for the extension (.lab for the LAB directory, .mp3 for the MP3 directory). The .lab files must be in the format used in Chris Harte's Beatles annotations.

Example usage:

    ./train -o scratch.pkl -m '/home/andreas/music/The Beatles' -l '/home/andreas/data/beatles_annotations/keylab/the_beatles' --limit 4

Inspecting the model
--------------------

Once the model has been trained (or using the pre-trained model that comes with this package), it can be inspected using the `inspect` tool:

    ./inspect -m test.pkl

Example output:

    C major matrix:
         C => C     : 0.028
         A => A     : 0.026
      D, G => D, G  : 0.023
         C => C#    : 0.017
         C => E, C  : 0.011
         D => D, G  : 0.011
         F => F     : 0.011
         C => C, G  : 0.009
         C => B     : 0.009
     D#, C => E, C  : 0.009
      G, C => E, C  : 0.009
        C# => C     : 0.009
      A, D => D     : 0.009
      F, G => C     : 0.009
      B, G => B, G  : 0.009
      E, A => E, A  : 0.009
     F, A# => C     : 0.009
         C => D#, C : 0.006
         C => C, E  : 0.006
         C => C, G# : 0.006
      E, C => C     : 0.006

    C minor matrix:
         C => C     : 0.024
        A# => A#    : 0.024
         F => G#, F : 0.018
     G#, F => F     : 0.018
     G#, F => G#, F : 0.018
     C, D# => A#, G : 0.012
     F, A# => D, A# : 0.012
         C => F, C  : 0.006
         C => G, C  : 0.006
         C => D#    : 0.006
         C => G#, F : 0.006
         C => F, A# : 0.006
         C => B     : 0.006
      D, C => F, D  : 0.006
     D#, C => G#, A#: 0.006
      F, C => F, C  : 0.006
      F, C => G, C  : 0.006
      G, C => F     : 0.006
      G, C => D#, G : 0.006
     A#, C => C     : 0.006
     A#, C => D#, C : 0.006


Classifying new examples
------------------------

Using a trained model, invoke the `test` tool on an MP3 file to detect its key.

    ./test -m test.pkl /home/andreas/music/The\ Beatles/With_the_Beatles/09_Hold_Me_Tight.mp3

If the model has been trained with good data, this command should output

    <MajorKey: F>


Evaluating the accuracy of the algorithm
----------------------------------------

The `evaluate` tool can be used to check the performance of the algorithm.


Getting documentation
---------------------

All four tools come with documentation about the available command line flags. To see this documentation, run

    ./NAME_OF_TOOL -h


Using the provided model
------------------------

This package comes with a model that has already been trained with 151 Beatles songs, to an accuracy of 75% (cross-validated on the same Beatles tracks). To use this model instead of having to train your own model, invoke `train` and `inspect` without the `-m` flag. (The default value for `-m` is `model.pkl`, which is the name of the pre-trained model.)