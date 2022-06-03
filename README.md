# SceneExtractor-2022 - A tool to extract images from a mp4 file of a video lecture.

Authors and Maintainers:
* Jiaxi Li <jiaxili3@illinois.edu>
* Zhong, Ivan <ninghan2@illinois.edu>
* Lawrence Angrave <angrave@illinois.edu>

## Acknowledgement & Citation
Please acknowledge this git repository https://github.com/classtranscribe/scene-extractor and the ASEE2022 paper (in submission) if you find this project useful.

## Based on
The original similarity metric and frame sampling code is from ClassTranscribe,
https://github.com/classtranscribe/WebAPI/blob/1274d4ee7599ba5943d95929eb6a97f5f9a23454/PythonRpcServer/scenedetector.py

## Example use
Run SceneExtractor on a single file
```python
python main.py example.mp4
```

Run SceneExtractor for all files in a folder
```python
python main.py foldername
```
