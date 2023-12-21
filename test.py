

from core.utils.OnnxExtractor import OnnxExtractor


if __name__ =="__main__":
    # Step 1, Create an Extractor instance
    extractor = OnnxExtractor()
    # Step 2, Extract the face data from image to json file
    data=extractor.extract(filename="test/sutaner_face.jpg",savepath="test/sutaner_face.json")
    # [Optional] Step 3, Print face data to the console
    print(data)