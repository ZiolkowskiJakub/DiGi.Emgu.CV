#### [DiGi\.Emgu\.CV](DiGi.Emgu.CV.Overview.md 'DiGi\.Emgu\.CV\.Overview')

## DiGi\.Emgu\.CV Namespace
### Classes

<a name='DiGi.Emgu.CV.Convert'></a>

## Convert Class

```csharp
public static class Convert
```

Inheritance [System\.Object](https://learn.microsoft.com/en-us/dotnet/api/system.object 'System\.Object') → Convert
### Methods

<a name='DiGi.Emgu.CV.Convert.ToEmguCV(thisSystem.Drawing.Color)'></a>

## Convert\.ToEmguCV\(this Color\) Method

Converts a [System\.Drawing\.Color](https://learn.microsoft.com/en-us/dotnet/api/system.drawing.color 'System\.Drawing\.Color') to an [Emgu\.CV\.Structure\.MCvScalar](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.structure.mcvscalar 'Emgu\.CV\.Structure\.MCvScalar')\.

```csharp
public static MCvScalar ToEmguCV(this System.Drawing.Color color);
```
#### Parameters

<a name='DiGi.Emgu.CV.Convert.ToEmguCV(thisSystem.Drawing.Color).color'></a>

`color` [System\.Drawing\.Color](https://learn.microsoft.com/en-us/dotnet/api/system.drawing.color 'System\.Drawing\.Color')

The color value to convert\.

#### Returns
[Emgu\.CV\.Structure\.MCvScalar](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.structure.mcvscalar 'Emgu\.CV\.Structure\.MCvScalar')  
An [Emgu\.CV\.Structure\.MCvScalar](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.structure.mcvscalar 'Emgu\.CV\.Structure\.MCvScalar') representing the BGR values of the specified color\.

<a name='DiGi.Emgu.CV.Query'></a>

## Query Class

```csharp
public static class Query
```

Inheritance [System\.Object](https://learn.microsoft.com/en-us/dotnet/api/system.object 'System\.Object') → Query
### Methods

<a name='DiGi.Emgu.CV.Query.AverageColorSimilarity(thisMat,Mat)'></a>

## Query\.AverageColorSimilarity\(this Mat, Mat\) Method

Calculates the average color similarity between two images, automatically selecting the GPU or CPU implementation based on CUDA availability\.

```csharp
public static double AverageColorSimilarity(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.AverageColorSimilarity(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first image matrix\.

<a name='DiGi.Emgu.CV.Query.AverageColorSimilarity(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second image matrix\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The Euclidean distance between the average colors of the two images, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input is null\.

<a name='DiGi.Emgu.CV.Query.AverageColorSimilarity_CPU(thisMat,Mat)'></a>

## Query\.AverageColorSimilarity\_CPU\(this Mat, Mat\) Method

Calculates the average color similarity between two images using the CPU implementation\.

```csharp
public static double AverageColorSimilarity_CPU(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.AverageColorSimilarity_CPU(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first image matrix\.

<a name='DiGi.Emgu.CV.Query.AverageColorSimilarity_CPU(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second image matrix\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The Euclidean distance between the average colors of the two images, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input is null\.

<a name='DiGi.Emgu.CV.Query.AverageColorSimilarity_GPU(Mat,Mat)'></a>

## Query\.AverageColorSimilarity\_GPU\(Mat, Mat\) Method

Calculates the average color similarity between two images using the GPU implementation for acceleration\.

```csharp
public static double AverageColorSimilarity_GPU(Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.AverageColorSimilarity_GPU(Mat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first image matrix\.

<a name='DiGi.Emgu.CV.Query.AverageColorSimilarity_GPU(Mat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second image matrix\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The Euclidean distance between the average colors of the two images, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input is null or CUDA is unavailable\.

<a name='DiGi.Emgu.CV.Query.AverageGrayMask(thisMat)'></a>

## Query\.AverageGrayMask\(this Mat\) Method

Generates a binary mask based on the average gray intensity of the provided image, automatically selecting between GPU and CPU implementations\.

```csharp
public static bool[]? AverageGrayMask(this Mat? mat);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.AverageGrayMask(thisMat).mat'></a>

`mat` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The input image matrix to process\.

#### Returns
[System\.Boolean](https://learn.microsoft.com/en-us/dotnet/api/system.boolean 'System\.Boolean')[\[\]](https://learn.microsoft.com/en-us/dotnet/api/system.array 'System\.Array')  
A boolean array representing the mask where true indicates pixels above the mean intensity, or null if the input is null\.

<a name='DiGi.Emgu.CV.Query.AverageGrayMask_CPU(thisMat)'></a>

## Query\.AverageGrayMask\_CPU\(this Mat\) Method

Generates a binary mask based on the average gray intensity of the provided image using CPU processing\.

```csharp
public static bool[]? AverageGrayMask_CPU(this Mat? mat);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.AverageGrayMask_CPU(thisMat).mat'></a>

`mat` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The input image matrix to process\.

#### Returns
[System\.Boolean](https://learn.microsoft.com/en-us/dotnet/api/system.boolean 'System\.Boolean')[\[\]](https://learn.microsoft.com/en-us/dotnet/api/system.array 'System\.Array')  
A boolean array representing the mask where true indicates pixels above the mean intensity, or null if the input is null\.

<a name='DiGi.Emgu.CV.Query.AverageGrayMask_GPU(Mat)'></a>

## Query\.AverageGrayMask\_GPU\(Mat\) Method

Generates a binary mask based on the average gray intensity of the provided image using GPU acceleration via CUDA\.

```csharp
public static bool[]? AverageGrayMask_GPU(Mat? mat);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.AverageGrayMask_GPU(Mat).mat'></a>

`mat` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The input image matrix to process\.

#### Returns
[System\.Boolean](https://learn.microsoft.com/en-us/dotnet/api/system.boolean 'System\.Boolean')[\[\]](https://learn.microsoft.com/en-us/dotnet/api/system.array 'System\.Array')  
A boolean array representing the mask where true indicates pixels above the mean intensity, or null if the input is null or CUDA is unavailable\.

<a name='DiGi.Emgu.CV.Query.ColorDistributionShift(thisMat,Mat)'></a>

## Query\.ColorDistributionShift\(this Mat, Mat\) Method

Calculates the color distribution shift between two images, automatically selecting between GPU and CPU implementations based on CUDA availability\.

```csharp
public static double ColorDistributionShift(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.ColorDistributionShift(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first image matrix\.

<a name='DiGi.Emgu.CV.Query.ColorDistributionShift(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second image matrix\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The Euclidean distance between the mean colors of the two images, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if inputs are invalid\.

<a name='DiGi.Emgu.CV.Query.ColorDistributionShift_CPU(thisMat,Mat)'></a>

## Query\.ColorDistributionShift\_CPU\(this Mat, Mat\) Method

Calculates the color distribution shift between two images using CPU processing\.

```csharp
public static double ColorDistributionShift_CPU(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.ColorDistributionShift_CPU(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first image matrix\.

<a name='DiGi.Emgu.CV.Query.ColorDistributionShift_CPU(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second image matrix\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The Euclidean distance between the mean colors of the two images, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input is null\.

<a name='DiGi.Emgu.CV.Query.ColorDistributionShift_GPU(Mat,Mat)'></a>

## Query\.ColorDistributionShift\_GPU\(Mat, Mat\) Method

Calculates the color distribution shift between two images using GPU acceleration via CUDA\.

```csharp
public static double ColorDistributionShift_GPU(Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.ColorDistributionShift_GPU(Mat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first image matrix\.

<a name='DiGi.Emgu.CV.Query.ColorDistributionShift_GPU(Mat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second image matrix\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The Euclidean distance between the mean colors of the two images, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input is null or CUDA is unavailable\.

<a name='DiGi.Emgu.CV.Query.ColorHistogramFactor(thisMat,Mat)'></a>

## Query\.ColorHistogramFactor\(this Mat, Mat\) Method

Calculates the color histogram similarity factor between two images, automatically selecting 
between GPU and CPU implementations based on CUDA availability\.

```csharp
public static double ColorHistogramFactor(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.ColorHistogramFactor(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first image matrix\.

<a name='DiGi.Emgu.CV.Query.ColorHistogramFactor(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second image matrix\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
A double value representing the correlation similarity between the two histograms\.

<a name='DiGi.Emgu.CV.Query.ColorHistogramFactor_CPU(thisMat,Mat)'></a>

## Query\.ColorHistogramFactor\_CPU\(this Mat, Mat\) Method

Calculates the color histogram similarity factor between two images using the CPU\.

```csharp
public static double ColorHistogramFactor_CPU(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.ColorHistogramFactor_CPU(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first image matrix\.

<a name='DiGi.Emgu.CV.Query.ColorHistogramFactor_CPU(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second image matrix\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
A double value representing the correlation similarity, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input matrix is null\.

<a name='DiGi.Emgu.CV.Query.ColorHistogramFactor_GPU(thisMat,Mat)'></a>

## Query\.ColorHistogramFactor\_GPU\(this Mat, Mat\) Method

Calculates the color histogram similarity factor between two images using the GPU via CUDA\.

```csharp
public static double ColorHistogramFactor_GPU(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.ColorHistogramFactor_GPU(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first image matrix\.

<a name='DiGi.Emgu.CV.Query.ColorHistogramFactor_GPU(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second image matrix\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
A double value representing the correlation similarity, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input matrix is null or CUDA is not available\.

<a name='DiGi.Emgu.CV.Query.Fill(thisMat,DiGi.Geometry.Planar.Interfaces.IPolygonal2D,System.Drawing.Color,bool)'></a>

## Query\.Fill\(this Mat, IPolygonal2D, Color, bool\) Method

Fills a polygonal area within the specified image with a given color\.

```csharp
public static Mat? Fill(this Mat? mat, DiGi.Geometry.Planar.Interfaces.IPolygonal2D? polygonal2D, System.Drawing.Color color, bool invert=false);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.Fill(thisMat,DiGi.Geometry.Planar.Interfaces.IPolygonal2D,System.Drawing.Color,bool).mat'></a>

`mat` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The source image to be filled\.

<a name='DiGi.Emgu.CV.Query.Fill(thisMat,DiGi.Geometry.Planar.Interfaces.IPolygonal2D,System.Drawing.Color,bool).polygonal2D'></a>

`polygonal2D` [DiGi\.Geometry\.Planar\.Interfaces\.IPolygonal2D](https://learn.microsoft.com/en-us/dotnet/api/digi.geometry.planar.interfaces.ipolygonal2d 'DiGi\.Geometry\.Planar\.Interfaces\.IPolygonal2D')

The polygonal shape defining the area to fill\.

<a name='DiGi.Emgu.CV.Query.Fill(thisMat,DiGi.Geometry.Planar.Interfaces.IPolygonal2D,System.Drawing.Color,bool).color'></a>

`color` [System\.Drawing\.Color](https://learn.microsoft.com/en-us/dotnet/api/system.drawing.color 'System\.Drawing\.Color')

The color used for filling\.

<a name='DiGi.Emgu.CV.Query.Fill(thisMat,DiGi.Geometry.Planar.Interfaces.IPolygonal2D,System.Drawing.Color,bool).invert'></a>

`invert` [System\.Boolean](https://learn.microsoft.com/en-us/dotnet/api/system.boolean 'System\.Boolean')

A flag indicating whether to invert the mask, filling the area outside the polygon instead of inside\.

#### Returns
[Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')  
A new image with the filled area, or null if the source image is null or the polygonal shape contains fewer than three valid points\.

<a name='DiGi.Emgu.CV.Query.Fill(thisMat,System.Drawing.Point[],MCvScalar,bool)'></a>

## Query\.Fill\(this Mat, Point\[\], MCvScalar, bool\) Method

Fills a convex polygonal area defined by an array of points within the specified image with a given MCvScalar color\.

```csharp
public static Mat? Fill(this Mat? mat, System.Drawing.Point[]? points, MCvScalar mCvScalar, bool invert=false);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.Fill(thisMat,System.Drawing.Point[],MCvScalar,bool).mat'></a>

`mat` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The source image to be filled\.

<a name='DiGi.Emgu.CV.Query.Fill(thisMat,System.Drawing.Point[],MCvScalar,bool).points'></a>

`points` [System\.Drawing\.Point](https://learn.microsoft.com/en-us/dotnet/api/system.drawing.point 'System\.Drawing\.Point')[\[\]](https://learn.microsoft.com/en-us/dotnet/api/system.array 'System\.Array')

An array of points defining the vertices of the polygon\.

<a name='DiGi.Emgu.CV.Query.Fill(thisMat,System.Drawing.Point[],MCvScalar,bool).mCvScalar'></a>

`mCvScalar` [Emgu\.CV\.Structure\.MCvScalar](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.structure.mcvscalar 'Emgu\.CV\.Structure\.MCvScalar')

The Emgu CV scalar color used for filling\.

<a name='DiGi.Emgu.CV.Query.Fill(thisMat,System.Drawing.Point[],MCvScalar,bool).invert'></a>

`invert` [System\.Boolean](https://learn.microsoft.com/en-us/dotnet/api/system.boolean 'System\.Boolean')

A flag indicating whether to invert the mask, filling the area outside the polygon instead of inside\.

#### Returns
[Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')  
A new image with the filled area, or null if the source image is null or the points array contains fewer than three points\.

<a name='DiGi.Emgu.CV.Query.GrayHistogramFactor(thisMat,Mat)'></a>

## Query\.GrayHistogramFactor\(this Mat, Mat\) Method

Calculates the correlation factor between the grayscale histograms of two image matrices\.

```csharp
public static double GrayHistogramFactor(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.GrayHistogramFactor(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first input image matrix\.

<a name='DiGi.Emgu.CV.Query.GrayHistogramFactor(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second input image matrix\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The histogram correlation value as a double, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input matrix is null\.

<a name='DiGi.Emgu.CV.Query.HammingDistance(thisMat,Mat)'></a>

## Query\.HammingDistance\(this Mat, Mat\) Method

Calculates the Hamming distance between two matrices based on their average gray masks\.

```csharp
public static int HammingDistance(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.HammingDistance(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first input matrix\.

<a name='DiGi.Emgu.CV.Query.HammingDistance(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second input matrix\.

#### Returns
[System\.Int32](https://learn.microsoft.com/en-us/dotnet/api/system.int32 'System\.Int32')  
The Hamming distance as an integer, or \-1 if either input matrix is null\.

<a name='DiGi.Emgu.CV.Query.HistogramCorrelation(thisMat,Mat,bool)'></a>

## Query\.HistogramCorrelation\(this Mat, Mat, bool\) Method

Calculates the correlation between the hue histograms of two images\.

```csharp
public static double HistogramCorrelation(this Mat? mat_1, Mat? mat_2, bool accumulate);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.HistogramCorrelation(thisMat,Mat,bool).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first input image matrix\.

<a name='DiGi.Emgu.CV.Query.HistogramCorrelation(thisMat,Mat,bool).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second input image matrix\.

<a name='DiGi.Emgu.CV.Query.HistogramCorrelation(thisMat,Mat,bool).accumulate'></a>

`accumulate` [System\.Boolean](https://learn.microsoft.com/en-us/dotnet/api/system.boolean 'System\.Boolean')

A flag indicating whether to accumulate the histogram\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The correlation value between the two histograms, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input matrix is null\.

<a name='DiGi.Emgu.CV.Query.LaplacianFactors(thisMat,double,double)'></a>

## Query\.LaplacianFactors\(this Mat, double, double\) Method

Calculates the mean and standard deviation of the Laplacian of the specified matrix\.

```csharp
public static void LaplacianFactors(this Mat? mat, out double mean, out double standardDeviation);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.LaplacianFactors(thisMat,double,double).mat'></a>

`mat` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The input matrix to process\.

<a name='DiGi.Emgu.CV.Query.LaplacianFactors(thisMat,double,double).mean'></a>

`mean` [System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')

When this method returns, contains the calculated mean value\.

<a name='DiGi.Emgu.CV.Query.LaplacianFactors(thisMat,double,double).standardDeviation'></a>

`standardDeviation` [System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')

When this method returns, contains the calculated standard deviation value\.

<a name='DiGi.Emgu.CV.Query.LaplacianFactors(thisMat,Mat,double,double)'></a>

## Query\.LaplacianFactors\(this Mat, Mat, double, double\) Method

Calculates the ratio of Laplacian factors between two specified matrices\.

```csharp
public static void LaplacianFactors(this Mat? mat_1, Mat? mat_2, out double mean, out double standardDeviation);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.LaplacianFactors(thisMat,Mat,double,double).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first input matrix\.

<a name='DiGi.Emgu.CV.Query.LaplacianFactors(thisMat,Mat,double,double).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second input matrix used as the divisor for the ratio\.

<a name='DiGi.Emgu.CV.Query.LaplacianFactors(thisMat,Mat,double,double).mean'></a>

`mean` [System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')

When this method returns, contains the ratio of the means of the two matrices\.

<a name='DiGi.Emgu.CV.Query.LaplacianFactors(thisMat,Mat,double,double).standardDeviation'></a>

`standardDeviation` [System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')

When this method returns, contains the ratio of the standard deviations of the two matrices\.

<a name='DiGi.Emgu.CV.Query.OpticalFlowAverageMagnitude(Mat,Mat)'></a>

## Query\.OpticalFlowAverageMagnitude\(Mat, Mat\) Method

Calculates the average magnitude of optical flow between two images, automatically selecting 
between GPU and CPU implementations based on CUDA availability\.

```csharp
public static double OpticalFlowAverageMagnitude(Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.OpticalFlowAverageMagnitude(Mat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first input image\.

<a name='DiGi.Emgu.CV.Query.OpticalFlowAverageMagnitude(Mat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second input image\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The average magnitude of the optical flow vectors\.

<a name='DiGi.Emgu.CV.Query.OpticalFlowAverageMagnitude_CPU(Mat,Mat)'></a>

## Query\.OpticalFlowAverageMagnitude\_CPU\(Mat, Mat\) Method

Calculates the average magnitude of optical flow between two images using the CPU\-based 
Farneback method\.

```csharp
public static double OpticalFlowAverageMagnitude_CPU(Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.OpticalFlowAverageMagnitude_CPU(Mat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first input image\.

<a name='DiGi.Emgu.CV.Query.OpticalFlowAverageMagnitude_CPU(Mat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second input image\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The average magnitude of the optical flow vectors, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input image is null\.

<a name='DiGi.Emgu.CV.Query.OpticalFlowAverageMagnitude_GPU(Mat,Mat)'></a>

## Query\.OpticalFlowAverageMagnitude\_GPU\(Mat, Mat\) Method

Calculates the average magnitude of optical flow between two images using the GPU\-accelerated 
Farneback method via CUDA\.

```csharp
public static double OpticalFlowAverageMagnitude_GPU(Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.OpticalFlowAverageMagnitude_GPU(Mat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first input image\.

<a name='DiGi.Emgu.CV.Query.OpticalFlowAverageMagnitude_GPU(Mat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second input image\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The average magnitude of the optical flow vectors, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input image is null or CUDA is not available\.

<a name='DiGi.Emgu.CV.Query.ORBFeatureMatchingFactor(thisMat,Mat)'></a>

## Query\.ORBFeatureMatchingFactor\(this Mat, Mat\) Method

Calculates the ORB feature matching factor between two images, automatically selecting 
between CPU and GPU implementations based on CUDA availability\.

```csharp
public static double ORBFeatureMatchingFactor(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.ORBFeatureMatchingFactor(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first image matrix to compare\.

<a name='DiGi.Emgu.CV.Query.ORBFeatureMatchingFactor(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second image matrix to compare\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The average distance of the feature matches; a lower value indicates a better match\. Returns [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if matching cannot be performed\.

<a name='DiGi.Emgu.CV.Query.ORBFeatureMatchingFactor_CPU(thisMat,Mat)'></a>

## Query\.ORBFeatureMatchingFactor\_CPU\(this Mat, Mat\) Method

Calculates the ORB feature matching factor between two images using the CPU\.

```csharp
public static double ORBFeatureMatchingFactor_CPU(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.ORBFeatureMatchingFactor_CPU(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first image matrix to compare\.

<a name='DiGi.Emgu.CV.Query.ORBFeatureMatchingFactor_CPU(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second image matrix to compare\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The average distance of the feature matches; a lower value indicates a better match\. Returns [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input is null or no descriptors are found\.

<a name='DiGi.Emgu.CV.Query.ORBFeatureMatchingFactor_GPU(thisMat,Mat)'></a>

## Query\.ORBFeatureMatchingFactor\_GPU\(this Mat, Mat\) Method

Calculates the ORB feature matching factor between two images using the GPU via CUDA\.

```csharp
public static double ORBFeatureMatchingFactor_GPU(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.ORBFeatureMatchingFactor_GPU(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first image matrix to compare\.

<a name='DiGi.Emgu.CV.Query.ORBFeatureMatchingFactor_GPU(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second image matrix to compare\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
The average distance of the feature matches; a lower value indicates a better match\. Returns [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if CUDA is unavailable, inputs are null, or no matches are found\.

<a name='DiGi.Emgu.CV.Query.ShapeComparisonFactor(thisMat,Mat,double,double)'></a>

## Query\.ShapeComparisonFactor\(this Mat, Mat, double, double\) Method

Calculates a shape comparison factor between two image matrices using Canny edge detection and Hu Moments\.

```csharp
public static double ShapeComparisonFactor(this Mat? mat_1, Mat? mat_2, double threshold_1=100.0, double threshold_2=200.0);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.ShapeComparisonFactor(thisMat,Mat,double,double).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first source image matrix\.

<a name='DiGi.Emgu.CV.Query.ShapeComparisonFactor(thisMat,Mat,double,double).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second source image matrix to compare against the first\.

<a name='DiGi.Emgu.CV.Query.ShapeComparisonFactor(thisMat,Mat,double,double).threshold_1'></a>

`threshold_1` [System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')

The lower threshold for the Canny edge detection algorithm\.

<a name='DiGi.Emgu.CV.Query.ShapeComparisonFactor(thisMat,Mat,double,double).threshold_2'></a>

`threshold_2` [System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')

The upper threshold for the Canny edge detection algorithm\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
A double value representing the sum of absolute differences between the Hu Moments of the primary contours; returns [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input matrix is null or no contours are detected in either image\.

<a name='DiGi.Emgu.CV.Query.StructuralSimilarityIndex_AbsoluteDifference(thisMat,Mat)'></a>

## Query\.StructuralSimilarityIndex\_AbsoluteDifference\(this Mat, Mat\) Method

Calculates a similarity index between two matrices based on the absolute difference of their grayscale representations\.

```csharp
public static double StructuralSimilarityIndex_AbsoluteDifference(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.StructuralSimilarityIndex_AbsoluteDifference(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first input matrix\.

<a name='DiGi.Emgu.CV.Query.StructuralSimilarityIndex_AbsoluteDifference(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second input matrix\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
A double value representing the similarity index, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input matrix is null\.

<a name='DiGi.Emgu.CV.Query.StructuralSimilarityIndex_MatchTemplate(thisMat,Mat)'></a>

## Query\.StructuralSimilarityIndex\_MatchTemplate\(this Mat, Mat\) Method

Calculates a similarity index between two matrices using template matching with normalized cross\-correlation\.

```csharp
public static double StructuralSimilarityIndex_MatchTemplate(this Mat? mat_1, Mat? mat_2);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.StructuralSimilarityIndex_MatchTemplate(thisMat,Mat).mat_1'></a>

`mat_1` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The first input matrix\.

<a name='DiGi.Emgu.CV.Query.StructuralSimilarityIndex_MatchTemplate(thisMat,Mat).mat_2'></a>

`mat_2` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The second input matrix\.

#### Returns
[System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')  
A double value representing the similarity index, or [System\.Double\.NaN](https://learn.microsoft.com/en-us/dotnet/api/system.double.nan 'System\.Double\.NaN') if either input matrix is null\.

<a name='DiGi.Emgu.CV.Query.TryMatchLocation(thisMat,Mat,double,double,System.Drawing.Point,System.Drawing.Point)'></a>

## Query\.TryMatchLocation\(this Mat, Mat, double, double, Point, Point\) Method

Attempts to match a template image within a target image using normalized cross\-correlation and retrieves the minimum and maximum values and their locations\.

```csharp
public static bool TryMatchLocation(this Mat? mat_Target, Mat? mat_Template, out double minValue, out double maxValue, out System.Drawing.Point minPoint, out System.Drawing.Point maxPoint);
```
#### Parameters

<a name='DiGi.Emgu.CV.Query.TryMatchLocation(thisMat,Mat,double,double,System.Drawing.Point,System.Drawing.Point).mat_Target'></a>

`mat_Target` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The source image in which to search for the template\.

<a name='DiGi.Emgu.CV.Query.TryMatchLocation(thisMat,Mat,double,double,System.Drawing.Point,System.Drawing.Point).mat_Template'></a>

`mat_Template` [Emgu\.CV\.Mat](https://learn.microsoft.com/en-us/dotnet/api/emgu.cv.mat 'Emgu\.CV\.Mat')

The template image to be matched\.

<a name='DiGi.Emgu.CV.Query.TryMatchLocation(thisMat,Mat,double,double,System.Drawing.Point,System.Drawing.Point).minValue'></a>

`minValue` [System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')

When this method returns, contains the minimum value of the match result\.

<a name='DiGi.Emgu.CV.Query.TryMatchLocation(thisMat,Mat,double,double,System.Drawing.Point,System.Drawing.Point).maxValue'></a>

`maxValue` [System\.Double](https://learn.microsoft.com/en-us/dotnet/api/system.double 'System\.Double')

When this method returns, contains the maximum value of the match result\.

<a name='DiGi.Emgu.CV.Query.TryMatchLocation(thisMat,Mat,double,double,System.Drawing.Point,System.Drawing.Point).minPoint'></a>

`minPoint` [System\.Drawing\.Point](https://learn.microsoft.com/en-us/dotnet/api/system.drawing.point 'System\.Drawing\.Point')

When this method returns, contains the location of the minimum value in the match result\.

<a name='DiGi.Emgu.CV.Query.TryMatchLocation(thisMat,Mat,double,double,System.Drawing.Point,System.Drawing.Point).maxPoint'></a>

`maxPoint` [System\.Drawing\.Point](https://learn.microsoft.com/en-us/dotnet/api/system.drawing.point 'System\.Drawing\.Point')

When this method returns, contains the location of the maximum value in the match result\.

#### Returns
[System\.Boolean](https://learn.microsoft.com/en-us/dotnet/api/system.boolean 'System\.Boolean')  
True if the matching process was successful; otherwise, false\.