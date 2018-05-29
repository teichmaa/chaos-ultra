template <class T> struct Point2D {
  T x;
  T y;
  __device__ Point2D(const T x,const T y) : x(x), y(y){} 
  __device__ ~Point2D(){}
  //template <class T> friend __device__ Point2D operator+ (const Point2D<T>& a,const Point2D<T>& b);
}; 

//todo s temi operatory to tady mam nejake divne

__device__ Point2D<int> operator+(const Point2D<int>& a,const Point2D<int>& b) {
  return Point2D<int>(a.x + b.x, a.y + b.y);
} 
__device__ Point2D<int> operator-(const Point2D<int>& a,const Point2D<int>& b) {
  return Point2D<int>(a.x - b.x, a.y - b.y);
} 
__device__ Point2D<int> operator*(const Point2D<int>& a,const Point2D<int>& b) {
  return Point2D<int>(a.x * b.x, a.y * b.y);
} 
__device__ Point2D<int> operator/(const Point2D<int>& a,const Point2D<int>& b) {
  return Point2D<int>(a.x / b.x, a.y / b.y);
} 


__device__ Point2D<float> operator+(const Point2D<float>& a,const Point2D<float>& b) {
  return Point2D<float>(a.x + b.x, a.y + b.y);
} 
__device__ Point2D<float> operator-(const Point2D<float>& a,const Point2D<float>& b) {
  return Point2D<float>(a.x - b.x, a.y - b.y);
} 
__device__ Point2D<float> operator*(const Point2D<float>& a,const Point2D<float>& b) {
  return Point2D<float>(a.x * b.x, a.y * b.y);
} 
__device__ Point2D<float> operator/(const Point2D<float>& a,const Point2D<float>& b) {
  return Point2D<float>(a.x / b.x, a.y / b.y);
} 


__device__ Point2D<double> operator+(const Point2D<double>& a,const Point2D<double>& b) {
  return Point2D<double>(a.x + b.x, a.y + b.y);
} 
__device__ Point2D<double> operator-(const Point2D<double>& a,const Point2D<double>& b) {
  return Point2D<double>(a.x - b.x, a.y - b.y);
} 
__device__ Point2D<double> operator*(const Point2D<double>& a,const Point2D<double>& b) {
  return Point2D<double>(a.x * b.x, a.y * b.y);
} 
__device__ Point2D<double> operator/(const Point2D<double>& a,const Point2D<double>& b) {
  return Point2D<double>(a.x / b.x, a.y / b.y);
} 


__device__ Point2D<long long> operator+(const Point2D<long long>& a,const Point2D<long long>& b) {
  return Point2D<long long>(a.x + b.x, a.y + b.y);
} 
__device__ Point2D<long long> operator-(const Point2D<long long>& a,const Point2D<long long>& b) {
  return Point2D<long long>(a.x - b.x, a.y - b.y);
} 
__device__ Point2D<long long> operator*(const Point2D<long long>& a,const Point2D<long long>& b) {
  return Point2D<long long>(a.x * b.x, a.y * b.y);
} 
__device__ Point2D<long long> operator/(const Point2D<long long>& a,const Point2D<long long>& b) {
  return Point2D<long long>(a.x / b.x, a.y / b.y);
} 


__device__ Point2D<long> operator+(const Point2D<long>& a,const Point2D<long>& b) {
  return Point2D<long>(a.x + b.x, a.y + b.y);
} 
__device__ Point2D<long> operator-(const Point2D<long>& a,const Point2D<long>& b) {
  return Point2D<long>(a.x - b.x, a.y - b.y);
} 
__device__ Point2D<long> operator*(const Point2D<long>& a,const Point2D<long>& b) {
  return Point2D<long>(a.x * b.x, a.y * b.y);
} 
__device__ Point2D<long> operator/(const Point2D<long>& a,const Point2D<long>& b) {
  return Point2D<long>(a.x / b.x, a.y / b.y);
} 