# Object Detection

### INTRODUCTION

<p>This project uses a fast real-time object detector to identify and localize various objects present in a live feed through a webcam. This will help segregate objects of concern from other objects. The real-time detection feature of this detector can also help in the surveillance of multiple places at once.</p>

<p>In this project, two big worldwide problems are identified, and an attempt is made to propose a solution for them.</p> 
<p>Poor waste management contributes to climate change and air pollution and directly affects many ecosystems and species. Failing to segregate waste properly means that it will end up mixed in landfills. Waste items like food scraps, paper, and liquid waste can mix and decompose, releasing run-off into the soil and harmful gas into the atmosphere.</p>
<p>On the other side, Plastic bags cause many minor and major ecological and environmental issues. In 2002, India banned the production of plastic bags below 20 µm in thickness to prevent plastic bags from clogging the municipal drainage systems and to prevent the cows of India from ingesting plastic bags as they confuse them for food. However, enforcement remains a problem. The Ministry of Environment, Forest and Climate Change has also passed a regulation to ban all polythene bags less than 50 microns on 18 March 2016. Due to poor implementation of this regulation, regional authorities (states and municipal corporations), have had to implement their own regulation [source: Wikipedia].</p>

### TECHNOLOGIES & TOOLS USED
<ul>
  <li>Python 3.9</li>
  <li>OpenCV, Numpy</li>
  <li>PyCharm, Google Colab</li>
  <li>labelImg</li>
  <li>OIDv4_ToolKit</li>
</ul>

### ACTIVITY DIAGRAM

<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/8e9600f6-94dd-4bab-9f75-7abe23c72381"/>

### IMPLEMENTATION STEPS

<p>1)	Firstly, Images are gathered from Open Images dataset - 513 images of plastic bags, 800 images of bottles, and 800 images of tin cans (Note- Images are in jpg format only).</p>

<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/28cb2ed4-b9eb-4496-b992-e499a9f31c1e" height="40%" width="60%"/>
<p><b>Figure 1.</b> Images Collected.</p>

<p>2)	Preprocessing / Annotation is performed. A text file is generated for each image. These files contain the location(s) of object instances in the images together with their class identities. Files contain this information in YOLO format (class id, object centers    (x, y), object width, and object height). These numbers are normalized by the real width and height of the images respectively. Text files are generated using a tool – labelImg.</p>

<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/85a787f1-9408-4ac8-a0e7-723d48fb9096" height="40%" width="60%"/>
<p><b>Figure 2.</b> Annotated Images</p>

<p>3)	Training is done on Google’s colab. Online GPU is utilized to speed up the process. Further, the advantage of pre-trained weights is taken and weights are downloaded and tested after every 2000 iterations. Overall, 6000 iterations are performed i.e., approximately 9 hours of training. The darknet framework is utilized for training purposes which is created also created by one of the contributors to the YOLO algorithm – Joseph Redmon. This framework serves as a backbone or feature extractor. Images are split into a 7:3 ratio for training and validation.</p>

<img arc="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/dc576e6c-9316-4525-9e9e-83b0c202ca67" height="40%" width="60%"/>
<p><b>Figure 3.</b> Training</p>

<p>4)	Finally, the model is evaluated by using charts provided by the Darknet framework and tested over some real-time images/feed from the webcam.</p>


### TESTING & FINDING

<p>Firstly, Mean Average Precision(mAP) is used to evaluate model performance. The mean of average precision values is calculated over recall values from 0 to 1. It uses other sub-metrics such as Confusion Matrix, Intersection over Union or Jaccard Index, Recall, and Precision.</p>
<p>These values are computed by the Darknet framework after every 1000 iterations.</p>

<p>Secondly, YOLOv3 uses binary cross-entropy loss for each label and computes total loss to plot a chart of mAP and loss values for each iteration.</p>

<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/9ead8077-eb81-478c-8936-a2ab2439a9d7" height="60%" width="40%"/>

<p><b>Figure 4.</b> Chart showing loss and mAP after 3000 iterations.</p>

<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/e96afdf1-a479-4e60-897c-4b232dbbc771" height="40%" width="30%"/>
<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/a01e06fb-3ff2-446d-bfd8-36afc3e1b1d5" height="40%" width="30%"/>
<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/e84462d0-fa1d-40cf-85c2-05f6b396e848"height="40%" width="30%"/>
<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/a44c84e6-2f10-4938-a128-0bd4d8a9189e"height="40%" width="30%"/>
<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/15f6f6a5-1874-4735-8367-018ef2c2437a"height="40%" width="30%"/>
<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/a5bc5751-6ab5-4d8e-a5f3-23843a5e3191"height="40%" width="30%"/>
<p text-align="center;"><b>Figure 5.</b> Object detected in real-time (True Positives).</p>

<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/cc309b96-0d83-45dc-8f0f-eb769d868b46" height="40%" width="30%"/>
<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/160c53ed-63e2-4874-987f-e452e034ab8d" height="40%" width="30%"/>
<p text-align="center;"><b>Figure 6.</b> Object detected in real-time (True Negatives).</p>

<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/ee211ece-23bd-4aa3-8e46-3207cd416131" height="40%" width="30%"/>
<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/0f7cd2a0-0ba7-463a-80d1-e1926d6e6f18" height="40%" width="30%"/>
<p text-align="center;"><b>Figure 7.</b> Object detected in real-time (False Negatives).</p>

<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/f046456c-b9b5-48ec-9b52-23cd61c98497" height="40%" width="30%"/>
<img src="https://github.com/Prakhar-Verma39/Object-Detection/assets/103757447/078e22ba-4c12-4b84-80a2-83a719b7e125" height="40%" width="30%"/>
<p text-align="center;"><b>Figure 8.</b> Object detected in real-time (False Positives).</p>
