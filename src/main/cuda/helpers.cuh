#ifndef HELPERS
#define HELPERS

#define DEBUG_MODE
#ifdef DEBUG_MODE 
  #define ASSERT(x) assert(x)
#else 
  #define ASSERT(x) do {} while(0)
#endif

#ifndef CUDART_VERSION
  #error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 9000) //for cuda 9 and later, use __any_sync(__activemask(), predicate) instead, see Programming guide, B.13 for more details
  #define __ALL(predicate) __all_sync(__activemask(), predicate)
  #define __ANY(predicate) __any_sync(__activemask(), predicate)
#else
  #define __ALL(predicate) __all(predicate)
  #define __ANY(predicate) __any(predicate)
#endif


template <class T> struct Point {
  T x;
  T y;
  __device__ Point(){}
  __device__ Point(const T x,const T y) : x(x), y(y){} 
  __device__ Point(const T both) : x(both), y(both){} 
  __device__ ~Point(){}
  __device__ Point<T> operator+ (const Point<T>& b){
    return Point<T>(x+b.x, y+b.y);
  }
  __device__ Point<T> operator- (const Point<T>& b){
    return Point<T>(x-b.x, y-b.y);
  }
  __device__ Point<T> operator* (const Point<T>& b){
    return Point<T>(x*b.x, y*b.y);
  }
  __device__ Point<T> operator/ (const Point<T>& b){
    return Point<T>(x/b.x, y/b.y);
  }
  __device__ Point<T> operator% (const Point<T>& b){
    return Point<T>(x%b.x, y%b.y);
  }
    __device__ const Point<T> operator+ (const Point<T>& b) const{
    return Point<T>(x+b.x, y+b.y);
  }
  __device__ const Point<T> operator- (const Point<T>& b) const{
    return Point<T>(x-b.x, y-b.y);
  }
  __device__ const Point<T> operator* (const Point<T>& b) const{
    return Point<T>(x*b.x, y*b.y);
  }
  __device__ const Point<T> operator/ (const Point<T>& b) const{
    return Point<T>(x/b.x, y/b.y);
  }
  __device__ const Point<T> operator% (const Point<T>& b) const{
    return Point<T>(x%b.x, y%b.y);
  }
  __device__ T manhattanDistanceTo (const Point<T>& b) const{
    return abs(x-b.x)+abs(y-b.y);
  }
  __device__ T distanceTo (const Point<T>& b) const{
    return sqrt((x-b.x)*(x-b.x)+(y-b.y)*(y-b.y));
  }
  template <class S> __device__ Point<S> cast() {
    return Point<S>((S)x, (S)y);
  }
  template <class S> __device__ const Point<S> cast() const {
    return Point<S>((S)x, (S)y);
  }
}; 

template <class T> struct Rectangle {
  Point<T> left_bottom;
  Point<T> right_top;
  __device__ Rectangle(){}
  __device__ Rectangle(Point<T> lt, Point<T> rb) : left_bottom(lb), right_top(rt){}
  __device__ Rectangle(const T lbx,const T lby, const T rtx,const T rty) : left_bottom(Point<T>(lbx, lby)), right_top(Point<T>(rtx, rty)){} 
  __device__ ~Rectangle(){}
  __device__ Point<T> const getLeftTop(){ return Point<T>(left_bottom.x, right_top.y); }
  __device__ Point<T> const getRightBottom(){ return Point<T>(right_top.x, left_bottom.y); }
  __device__ Point<T> const size(){ return right_top - left_bottom;}
};

struct fov_result_t{
  float advisedSampleCount;
  bool isInsideFocusArea;

  __device__ fov_result_t(float advisedSampleCount, bool isInsideFocusArea)
    : advisedSampleCount(advisedSampleCount), isInsideFocusArea(isInsideFocusArea)
  {
  }
  __device__ fov_result_t(const fov_result_t & b){
    advisedSampleCount = b.advisedSampleCount;
    isInsideFocusArea = b.isInsideFocusArea;
  }
  __device__ fov_result_t() : advisedSampleCount(0), isInsideFocusArea(false) {}
};

struct pixel_info_t{
    /// The value of the fractal
    float value;
    /// How important the value is. Initially, it equals the number of samples takes, but may decrease over time
    float weight;
    bool isReused;
    float weightOfNewSamples;
    __device__ pixel_info_t(unsigned int value, float weight)
      : value(value), weight(weight), isReused(false), weightOfNewSamples(0)
    {
    }
    __device__ pixel_info_t(unsigned int value, unsigned int weight)
      : value(value), weight((float) weight), isReused(false), weightOfNewSamples(0)
    {
    }
    __device__ pixel_info_t(const pixel_info_t & b){
      value = b.value;
      weight = b.weight;
      isReused = b.isReused;
      weightOfNewSamples = b.weightOfNewSamples;
    }
    __device__ pixel_info_t() : value(0), weight(0), isReused(false), weightOfNewSamples(0) {}
};

struct rgba{
  char r;
  char g;
  char b;
  char a;
};

typedef struct color_t
{
  union
  {
    unsigned int intValue;
    struct rgba rgba;
  };
} color_t;
     


class ColorsRGBA{ 
public:
//when human read, it is abgr, because of endianity
//                                              aabbggrr
  static constexpr const unsigned int BLACK = 0xff000000;
  static constexpr const unsigned int WHITE = 0xffffffff;
  static constexpr const unsigned int PINK  = 0xffb469ff;
  static constexpr const unsigned int GOLD  = 0xff00d7ff;
  static constexpr const unsigned int YELLOW= 0xff00ffff;
  static constexpr const unsigned int BLUE  = 0xffff0000;
  static constexpr const unsigned int GREEN = 0xff00ff00;
  static constexpr const unsigned int RED   = 0xff0000ff;
}; 

#endif
