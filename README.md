# LiFS-implementation

Indoor localization is an topic of interest in research, as this problem is currently not satisfactorily solved. There are many systems and solutions for indoor localization, from using radio signal propagation, to using beacons scattered at known locations in the building, or QR codes to reset the error using smartphone’s camera. The spread of smartphones and existence of WiFi infrastructures in buildings motivates the development of localization solutions that are using the signal strength of Access Points for precise positioning, since this type of solution does not involve additional infrastructure costs. For this reason, we chose to focus on such solutions, considering that a solution that does not involve high initial setup costs has a good chance of being adopted on a large scale and having a strong impact.

A problem of this research area is the lack of source code for the solutions proposed in the literature. This makes it difficult for the research to advance, as in order to improve existing solutions or to compare with existing solutions, researchers are forced to implement each such solution from scratch. This is a considerable effort.

For this reason, the contribution of this project is to implement an existing localization system with impact and publish the source code for the benefit of the research community. The chosen system is LiFS, a localization system with good results in terms of accuracy, which uses a phone and the existing WiFi infrastructure in a building, and which aims to eliminate the cost associated with the system setup process, which consists of initial collection of a database of manually annotated WiFi footprints and locations.

The implemented system was evaluated on two data sets, representing different types of scenarios - an apartment in a residential building, and an office building. The system implementation was validated by comparing the system performance obtained in the two scenarios, with the performance obtained by the authors who proposed the solution.

## Project structure

There are 2 available data sets. The project was fully tested on the second database, on Scenario 1. 
