import socket
import sys
import json
import threading

import requests
import imghdr
import os

import tempfile
import shutil

import base64
from PIL import Image
from urllib.parse import urlparse
import cv2
import sewar
import numpy as np

class OllamaImageAnalyzer:

    def __init__(
        self,
        inputPort: int = 11433
    ):
        self.defaultBufferSize = 4096
        self.tempImageNameWithExtension = ""
        self.tempImageName = ""
        self.tempImageDirectory = ""
        self.analysis = []
        self.promptLengthLimit = 2000
        self.numPassesLimit = 10
        self.inspectImageQualityWidthMinimum = 1920
        self.inspectImageQualityHeightMinimum = 1080
        self.inspectImageQualityBlurMaximum = 100
        self.inspectImageQualityPSNRMinimum = 30.0
        self.inspectImageQualitySSIMMinimum = 0.8
    
        self.defaultNumPasses = 3
        self.defaultLoopLimit = 20
        self.defaultPrompt = "What is this an image of?"
        self.apiUrl = "http://0.0.0.0:11434/api/generate"
        #self.apiImageVisionModel = "llava:7b-v1.5-q6_K"
        
        #self.apiImageVisionModel = "sharegpt4v-7b_q5_k_m"
        self.apiImageVisionModel = "sharegpt4v-13b-q4_k_m"
        self.apiSummaryModel = "openhermes2.5-mistral"
        self.startServer(inputPort)


        self.passedNumPassesCheck = ""
        self.passedImageCheck = ""
        self.passedPromptCheck = ""

        self.passedResolutionCheck = ""
        self.passedBlurCheck = ""
        self.passedSSIMCheck = ""
        self.passedPSNRCheck = ""

        pass

    def parseArguments(
        self, 
        inputData: str
    ) -> None:
        # Split the data based on double line breaks
        dataParts = inputData.split('\r\n\r\n', 1)
        header = dataParts[0]
        jsonData = dataParts[1]

        if (len(jsonData) > 0):
            try:
                print(jsonData)
                arguments = json.loads(jsonData)
    
                return arguments
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return None

    def handleClient(
        self,
        client_socket, 
        client_address
    ):
        print(f"Connection established from {client_address}")
        #client_socket.settimeout(1)
        #sock.setblocking(False)
        #client.shutdown(socket.SHUT_WR)

        byteData = b""
        while True:
            try:
                chunk = client_socket.recv(self.defaultBufferSize)
                chunkLength = len(chunk)

                print("Received chunk. Type:", type(chunk), "Length:", str(chunkLength))

                byteData += chunk

                if (chunkLength < self.defaultBufferSize):
                    client_socket.shutdown(SHUT_WR)

            except:
                break
            


        decodedData = byteData.decode('utf-8')
        arguments = self.parseArguments(decodedData)
        print('start args received')
        print(arguments)
        print('end args received')
        
        if arguments is not None:
            print("Received arguments:")
            for key, value in arguments.items():

                print(f"{key}: {value}")   

        self.passedNumPassesCheck = self.__validateNumPasses(arguments["numPasses"])
        self.passedImageCheck = self.__validateImage(arguments["imageUrl"], arguments["imageContent"])
        self.passedPromptCheck = self.__validatePrompt(arguments["prompt"])

        print("passedNumPassesCheck: " + str(self.passedNumPassesCheck))
        print("passedImageCheck: " + str(self.passedImageCheck))
        print("passedPromptCheck: " + str(self.passedPromptCheck))

        if (
            (self.passedNumPassesCheck == True) and 
            (self.passedImageCheck == True) and
            (self.passedPromptCheck == True)
            #self.__validateGuidance(arguments.guidance)
        ):
            self.__runImageExpertsAnalysis()
            self.__runImageSummaryAnalysis()
        
            self.numPasses = 0
            self.prompt = ""

            if os.path.exists(self.tempImageDirectory):
                shutil.rmtree(self.tempImageDirectory)
            self.tempImageDirectory = ""
            self.tempImageNameWithExtension = ""
            self.tempImageName = ""
            self.tempImagePath = ""
            self.convertedTempImagePath = ""


        client_socket.close()
        print(f"Connection with {client_address} closed")

    def __runImageSummaryAnalysis(
        self
    ) -> None:
        prompt = """
        {{numPasses}} experts analyzed an image and provided their OBSERVATIONS. 
        Extrapolate from their combined OBSERVATIONS the common key elements

        OBSERVATIONS:
        """
        prompt.replace("{{numPasses}}", str(self.numPasses))

        for message in self.analysis:
            prompt += message
            prompt += "\n"

        payload = {
            "model": self.apiSummaryModel,
            "prompt": prompt,
            "stream": False
        }

        json_payload = json.dumps(payload)
        headers = {
            "Content-Type": "application/json"
        }   

        while True:
            try:
                response = requests.post(
                    self.apiUrl, 
                    data=json_payload, 
                    headers=headers
                )  

                response.raise_for_status()
                responseData = json.loads(response.content.decode('utf-8'))
                message = responseData["response"].strip()    
                messageLength = len(message)

                print("__runImageSummaryAnalysis messageLength was " + str(messageLength))
                
                if (messageLength > 0):
                    break

            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
                return False

        print(message)  

                
    def __runImageExpertsAnalysis(
        self
    ) -> None:
        imageContentBase64Encoded = self.__getBase64EncodedLocalImage(self.convertedTempImagePath)
    
        payload = {
            "model": self.apiImageVisionModel,
            "prompt": self.prompt,
            "images": [imageContentBase64Encoded],
            "stream": False
        }

        json_payload = json.dumps(payload)
        headers = {
            "Content-Type": "application/json"
        }   

        countPasses = 1
        self.analysis = []

        while countPasses <= self.numPasses:
            
            if (countPasses >= self.defaultLoopLimit):
                break
            
            try:
                response = requests.post(
                    self.apiUrl, 
                    data=json_payload, 
                    headers=headers
                )  

                response.raise_for_status()
                responseData = json.loads(response.content.decode('utf-8'))
              
                message = responseData["response"].strip()
                messageLength = len(message)

                print("__runImageExpertsAnalysis messageLength was " + str(messageLength))

                if (messageLength > 0):
                    self.analysis.append(message)

            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
                return False
              
            if (messageLength > 0):            
                countPasses += 1

        print(self.analysis)


            

        

    def __getBase64EncodedLocalImage(
        self,
        inputImagePath: str        
    ) -> str:
        with open(inputImagePath, "rb") as imageFileObject:
            # Read the binary data of the image file
            imageContentBinaryData = imageFileObject.read()
        imageContentBase64Encoded = base64.b64encode(imageContentBinaryData)
        imageContentBase64Encoded = imageContentBase64Encoded.decode('utf-8')

        output = imageContentBase64Encoded

        return output

    def __validateNumPasses(
        self,
        inputNumPasses: str
    ) -> bool:

        numPasses = self.__cleanNumPasses(inputNumPasses)
        print('numPasses is ' + str(numPasses))
        if ((numPasses <= 0) or (numPasses > self.numPassesLimit)):
            return False

        self.numPasses = numPasses

        return True

    def __cleanNumPasses(
        self,
        inputNumPasses: str
    ) -> int:

        output = inputNumPasses.strip()
        output = int(output)

        return output
        
    def __validatePrompt(
        self,
        inputPrompt: str
    ) -> bool:

        prompt = self.__cleanPrompt(inputPrompt)
        promptLength = len(prompt)

        if (promptLength == 0):
            prompt = self.defaultPrompt
        elif (promptLength > self.promptLengthLimit):
            return False

        self.prompt = prompt

        print(self.prompt)

        return True

    def __cleanPrompt(
        self,
        inputPrompt: str
    ) -> str:

        output = inputPrompt.strip()

        return output

    def __validateImage(
        self,
        inputImageUrl: str = "",
        inputImageContent: str = ""
    ) -> bool:

        imageUrl = self.__cleanImageUrl(inputImageUrl)
        imageContent = self.__cleanImageContent(inputImageContent)

        imageUrlLength = len(imageUrl)
        imageContentLength = len(imageContent)

        if ((imageUrlLength == 0) and (imageContentLength == 0)):
            return False

        if (imageUrlLength > 0):
            self.__validateImageUrl(imageUrl)
        elif (imageContentLength > 0):
            self.__validateImageContent(imageContent)

        return True

    def __cleanImageUrl(
        self,
        inputImageUrl: str
    ) -> str:

        output = inputImageUrl.strip()

        return output

    def __cleanImageContent(
        self,
        inputImageContent: bytes
    ) -> str:

        output = inputImageContent.strip()  
        output = base64.b64decode(output) 

        return output

    def __downloadImage(
        self,
        inputImageContent: bytes,
        inputImageExtension: str
    ) -> None:
        tempImageDirectory = tempfile.mkdtemp()        
        tempImageObject = tempfile.NamedTemporaryFile(dir=tempImageDirectory, suffix=inputImageExtension)
        tempImagePath = tempImageObject.name
        tempImageName = os.path.splitext(os.path.basename(tempImagePath))[0]
        tempImageNameWithExtension = os.path.basename(tempImagePath)
        
        print("tempImageDirectory: " + tempImageDirectory)
        print("tempImageName: " + tempImageName)
        print("tempImageNameWithExtension: " + tempImageNameWithExtension)
        print("tempImagePath: " + tempImagePath)

        # Write the content of the response to the temporary file
     
        with open(tempImagePath, 'wb') as file:
            file.write(inputImageContent)
            file.close()

        convertedTempImagePath = os.path.join(tempImageDirectory, tempImageName + "_converted.png")
        print(convertedTempImagePath)
        # Return the path to the temporary file and directory

        self.tempImageDirectory = tempImageDirectory
        self.tempImageNameWithExtension = tempImageNameWithExtension
        self.tempImageName = tempImageName
        self.tempImagePath = tempImagePath
        self.convertedTempImagePath = convertedTempImagePath
        
        with Image.open(tempImagePath) as img:
            img.save(convertedTempImagePath, format='PNG')
           
        self.__detectImageQuality()


    def __detectImageQualityResolution(
        
        self
    ) -> bool:
        output = True
        tempImage = Image.open(self.convertedTempImagePath)
        imageWidth = tempImage.width
        imageHeight = tempImage.height

        print("imageWidth: " + str(imageWidth))
        print("imageHeight: " + str(imageHeight))

        # Detect if image is less than 1920x1080 resolution
        if (imageWidth < self.inspectImageQualityWidthMinimum or imageHeight < self.inspectImageQualityHeightMinimum):
            output = False

        return output

    def __detectImageQualityBlur(
        self,
        inputTempImage: np.ndarray

    ) -> bool:
        output = True
        
        tempImageGray = cv2.cvtColor(
            inputTempImage, 
            cv2.COLOR_BGR2GRAY
        )
        blurValue = cv2.Laplacian(
            tempImageGray, 
            cv2.CV_64F
        ).var()
        
        print("blurValue: " + str(blurValue))

        if (blurValue <= self.inspectImageQualityBlurMaximum):
            output = False

        return output

    def __detectImageQualitySSIM(
        self,
        inputImageGray: list,
        inputImageReferenceGray: list
    ) -> bool: 
        output = True

        ssimIndex, _ = sewar.full_ref.ssim(
            inputImageGray, 
            inputImageReferenceGray
        )
        
        print("ssimIndex: " + str(ssimIndex))

        if (ssimIndex < self.inspectImageQualitySSIMMinimum):
            output = False

        return output

    def __detectImageQualityPSNR(
        self,
        inputImageGray: list,
        inputImageReferenceGray: list
    ) -> bool:
        output = True

        psnr = sewar.full_ref.psnr(
            inputImageGray, 
            inputImageReferenceGray
        )  

        print("psnr: " + str(psnr))
        print(str(type(psnr)))
        
        if (str(psnr) != "inf") and (psnr < self.inspectImageQualityPSNRMinimum):
            output = False

        return output  

    def __detectImageQuality(
        self
    ) -> None:
        tempImage = cv2.imread(self.convertedTempImagePath)
        tempImageReference = cv2.imread(self.convertedTempImagePath)
        
        tempImageGray = cv2.cvtColor(
            tempImage, 
            cv2.COLOR_BGR2GRAY
        )

        tempImageReferenceGray = cv2.cvtColor(
            tempImageReference, 
            cv2.COLOR_BGR2GRAY
        )

        print("tempImageGray: " + str(tempImageGray))
        print("tempImageReferenceGray: " + str(tempImageReferenceGray))

        self.passedResolutionCheck = self.__detectImageQualityResolution()
        self.passedBlurCheck = self.__detectImageQualityBlur(tempImage)
        self.passedSSIMCheck = self.__detectImageQualitySSIM(
            tempImageGray, 
            tempImageReferenceGray
        )
        self.passedPSNRCheck = self.__detectImageQualityPSNR(
            tempImageGray,
            tempImageReferenceGray
        )

        print("passedResolutionCheck: " + str(self.passedResolutionCheck))
        print("passedBlurCheck: " + str(self.passedBlurCheck))
        print("passedSSIMCheck: " + str(self.passedSSIMCheck))
        print("passedPSNRCheck: " + str(self.passedPSNRCheck))

        #sys.exit(0)

    def __validateImageUrl(
        self,
        inputImageUrl: str
    ) -> bool:

        imageUrl = inputImageUrl
        
        try:
            # Make a HEAD request to check if the URL is accessible
            response = requests.head(imageUrl)
            response.raise_for_status()

            # Check Content-Type header
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                return False

            # Download a small portion of the content (first 512 bytes) for inspection
            sampleContent = requests.get(imageUrl, stream=True).content[:512]
            print(type(sampleContent))
            # Check if the downloaded content is in a valid image format
            imageFormat = imghdr.what(None, h=sampleContent)

            # Check file extension
            parsedUrl = urlparse(imageUrl)
            fileExtension = os.path.splitext(parsedUrl.path)[1]

            print('__validateImageUrl file extension is ')
            print(fileExtension)
            print('end file extension')

            validExtensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
            if imageFormat is None or fileExtension.lower() not in validExtensions:
                return False

            imageContentResponse = requests.get(imageUrl)
            imageContent = imageContentResponse.content

            self.__downloadImage(imageContent, fileExtension)

            return True

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return False        

    def __validateImageContent(
        self,
        inputImageContent: bytes
    ) -> bool:
        imageContent = inputImageContent
        print(type(imageContent))
        imageFormat = imghdr.what(None, h=imageContent)

        # Check if the content is in a valid image format
        validExtensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        fileExtension = "." + imageFormat
        print('__validateImageContent file extension is ')
        print(fileExtension)
        print('end file extension')
        if imageFormat is None or fileExtension.lower() not in validExtensions:
            return False

        self.__downloadImage(imageContent, fileExtension)

        return True


    def startServer(
        self,
        inputPort: int
    ):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            server_socket.bind(('localhost', inputPort))
            server_socket.listen(5)
            print(f"Server listening on port {inputPort}...")

            while True:
                client_socket, client_address = server_socket.accept()
                client_handler = threading.Thread(target=self.handleClient, args=(client_socket, client_address))
                client_handler.start()

        except socket.error as e:
            print(f"Socket error: {e}")

        finally:
            server_socket.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ollama_image_analyzer.py <port>")
        sys.exit(1)

    port = int(sys.argv[1])
    OllamaImageAnalyzerObject = OllamaImageAnalyzer(port)
