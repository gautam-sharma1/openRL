# openRL



## Getting Started

```bash
pip install pygame
```

Install pytorch according to your OS ![https://pytorch.org]


## Files
**Agents.py** defines the class to make your environment. Play around with the Robot class and if impossible let's build a better environment.<br/>
**AI.py** has all the algorithms defined<br/>
**ReplayBuffer.py** defines the class to store all **(state ,action ,next_state, reward ,done)** transitions<br/>

<!--
## Heuristic 
There are numerous heuristics that can be used in path planning algorithms. Use what is suitable for your problem. 
```cpp
/*
Heuristic funtion. Should be modified according to the problem.
Current heuristic is (x1-x2)^2 + (y1-y2)^2
*/
double Heuristic(const PointI &a, const PointI &b)
{
    return 10 * std::sqrt(((std::get<0>(a) - std::get<0>(b)) ^ 2 + (std::get<1>(a) - std::get<1>(b)) ^ 2));
};
```
## Complexity
<img src="Astar_comp.png" width="700" height="200" /> <br/> 
(Source : http://www.ai.mit.edu/courses/6.034b/searchcomplex.pdf)
-->
<!--
## Contributing
This repository is in it's beginning stages. The goal of this project is to use the speed and efficiency of C++ along with its modern syntax to provide a simple interface for the user to test his/her algorithms. If you feel you are interested in contributing please send me an email as I am still in the process of finalizing a "how to contribute?" guidelines. Thank you :)
<!--
### Prerequisites
OpenCV <br/>
C++ 17 <br/>
cmake <br/>
<!--
### Installing
Installing OpenCV: <br/>
**[Install HomeBrew]**:
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
**[Install OpenCV]**: <br/>
```
brew install opencv
```
<!--
**[Install pkg-config]**:<br/>
```
brew install pkg-config
```
Clone the repo:
```
git clone https://github.com/gautam-sharma1/AStar-Search-ModernCpp.git

```
### Running
```
cd AStar-Search-ModernCpp
cd build
cmake .. && make
./main
```



## Built With

* [OpenCV](https://docs.opencv.org/3.4/) - Used for Visualization


## Authors

* **Gautam Sharma** - *Initial work* - [Github](https://github.com/gautam-sharma1)
* Please leave a star if you find this repo interesting!


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
-->



