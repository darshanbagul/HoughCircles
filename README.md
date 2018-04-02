# Hough Transform for Circle Detection

Hough Transforms is a feature extraction technique used for finding imperfect instances of objects belonging to a certain class of shapes, using a voting procedure. This voting procedure is carried out in parameter space from which object candidates are obtained as local maxima in accumulator space, which is constructed by the algorithm computing Hough Transform. One of the main features of Hough Transforms is to perform groupings of edge points into object candidates by performing an explicit voting procedure over a set of parameterized image objects. The simplest case of Hough transform is detecting straight lines, but it can be extended to identify variety of shapes, most commonly Ellipse or a Circle.

## Dependencies

```
  matplotlib==1.5.1
  cv2==2.4.13
  numpy==1.13.0
```

## Algorithm
![Image](https://github.com/darshanbagul/HoughCircles/blob/master/images/execution_flow.png)

## Result

### Input

![Image](https://github.com/darshanbagul/HoughCircles/blob/master/images/HoughCircles.jpg)

### Output

![Image](https://github.com/darshanbagul/HoughCircles/blob/master/images/result.png)
