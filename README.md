# Biometrics Face Anti-spoofing

## Face Anti-spoofing with 2D lip articulation analysis

  The idea is not just to make a binary classifier of two obvious visual lip conditions (open and closed), but make a word classifier under conditions of uncertainty: there are many words in every language that can be made up using same combinations of visems (visual counterparts of phonemes). This uncertainty can be fixed using a limited set of words, that can visually be distinguished from each other. And so in russian language that set of words can clearly be a set of numbers from 0 to 9 as their viseme combinations are unique within this set.
  
  Thus if the required combination of numbers is read and can be correctly classified then vitality/anti-spoofing test is positive (there is no attack on biometric system), otherwise session is considered an attack.
  
  This kind of anti-spoofing test can prevent such attempts of attack as photo-attacks, video-playback attacks and different types of masked attacks (with paper, stone powder and silicon 3D-masks).
  
### This repository contains python pieces of code for:
* raw data extraction out of video records of face
* normalization and feature creation
* data analysis, preprocessing and feature extraction
* test random forest classifier for observed articulation

### Still working on the rest...
