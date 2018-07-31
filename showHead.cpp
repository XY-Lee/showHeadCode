#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>
using namespace cv;
using namespace std;
// the minimum object size
int min_face_height = 50;
int min_face_width = 50;

# define HEAD_SHOW_HEIGHT 100
# define HEAD_SHOW_WIDTH 100

# define ANI_HEAD_MAX 29

// 所有動物頭的檔案名稱
const string gHeadName[29] = { "01.png", "02.png", "03.png", "04.png", "05.png", "06.png",
  "07.png", "08.png", "09.png", "10.png", "11.png", "12.png",
  "13.png", "14.png", "15.png", "16.png", "17.png", "18.png",
  "19.png", "20.png", "21.png", "22.png", "23.png", "24.png",
  "25.png", "26.png", "27.png", "28.png", "29.png" } ;

// 自定義的資料結構
class Img {
public :
  string name ; // 圖片檔名
  string path ; // 圖片路徑
  Mat img ; // opencv 資料結構

  Img() {} // Img() 

  Img( string n ) { // 傳入單個檔案名稱 讀取目前資料夾的那個檔案
    name = n ;
    path = "./" ;
    img = imread( path + n, CV_LOAD_IMAGE_UNCHANGED);
  } // Img()

  Img( string n, string inPath ) {  // 傳入單個檔案名稱與路徑 讀取路徑中的那個檔案
    name = n ;
    path = inPath ;
    img = imread( inPath + n, CV_LOAD_IMAGE_UNCHANGED);
  } // Img()
} ;

Img gAniHead[29] ; // 存放所有動物圖片

int cvAdd4cMat_q( Mat &dst, Mat &scr, double scale) {   // 將 圖dst 與 scr 以 透明度 scale 作融合
  if (dst.channels() != 3 || scr.channels() != 4) {   // 檢查兩張圖的資料大小是否一致
    return false;    
  } // if 
  if (scale < 0.01)    
    return false;    
  vector<Mat>scr_channels;    
  vector<Mat>dstt_channels;    
  split(scr, scr_channels);    
  split(dst, dstt_channels);    
  CV_Assert(scr_channels.size() == 4 && dstt_channels.size() == 3);    

  if (scale < 1) {    
    scr_channels[3] *= scale;    
    scale = 1;    
  } // if

  for (int i = 0; i < 3; i++) {    
    dstt_channels[i] = dstt_channels[i].mul(255.0 / scale - scr_channels[3], scale / 255.0);    
    dstt_channels[i] += scr_channels[i].mul(scr_channels[3], scale / 255.0);    
  } // for

  merge(dstt_channels, dst);    
  return true;    
} // cvAdd4cMat_q()

void ShowAnimalHead() { // 將 所有的 gAniHead 融合在一個視窗 隨後 show 出
  Scalar color = Scalar(255,255,255,255);
  Scalar fontColor = Scalar(0,0,255);
  Mat backMat( HEAD_SHOW_WIDTH * 6, HEAD_SHOW_HEIGHT * 5, CV_8UC4, color ) ;
  Mat temp ;
  Mat resizeImg ;
  for ( int i = 0 ; i < 6 ; i++ ) {
    for ( int j = 0 ; j < 5 && j + 5 * i < ANI_HEAD_MAX ; j++ ) {
      resize( gAniHead[j + 5 * i].img, resizeImg, Size( HEAD_SHOW_WIDTH, HEAD_SHOW_HEIGHT ) ) ;
      temp = backMat( Rect( j * HEAD_SHOW_WIDTH , i * HEAD_SHOW_HEIGHT, resizeImg.cols, resizeImg.rows ) );  //指定插入的大小和位置

      if ( !cvAdd4cMat_q( temp, resizeImg, 1.0 ) ) // 先試試看此方法有沒有成功將圖融合
        addWeighted( temp, 0, resizeImg, 1.0, 0, temp ); // 如果沒有成功 就換成 openCV 的內建 function
      
      // 以下是將圖片寫上文字
      stringstream temp ;
      temp << j + 5 * i ;
      string w ;
      temp >> w ;
      w = w + ":" + gAniHead[j + 5 * i].name ;
      putText( backMat, w, Point( j * HEAD_SHOW_WIDTH, i * HEAD_SHOW_HEIGHT + 15 ),  FONT_HERSHEY_DUPLEX, 0.6, fontColor ) ;

    } // for
  } // for

  namedWindow("Animal Head", 1); // 創造一個視窗 命名為 Animal Head
  imshow("Animal Head", backMat ) ; // 將剛剛融合成一張的動物圖 放到此視窗上
} // ShowAnimalHead()

void ShowDetectFace( Img mainImg, CvSeq *&faces ) { // 將偵測照片的頭用紅框框起來並show在螢幕上
  // Load image
  IplImage *image_detect = new IplImage( mainImg.img );
  string cascade_name = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml" ; // opencv 對於臉的分類的資料集
  // Load cascade
  // 用 haar分類器 這個方法來做人臉偵測 
  CvHaarClassifierCascade* classifier=(CvHaarClassifierCascade*)cvLoad(cascade_name.c_str(), 0, 0, 0); 
  if ( !classifier ) {
    cerr<<"ERROR: Could not load classifier cascade."<<endl;
    return ;
  } // if

  CvMemStorage *facesMemStorage = cvCreateMemStorage( 0 ) ;
  IplImage *tempFrame=cvCreateImage( cvSize( image_detect->width, image_detect->height ), IPL_DEPTH_8U, image_detect->nChannels );
  if ( image_detect->origin == IPL_ORIGIN_TL ) {
    cvCopy(image_detect, tempFrame, 0);
  } // if
  else{
    cvFlip( image_detect, tempFrame, 0 );
  } // else

  cvClearMemStorage(facesMemStorage);
  
  // 將辨識出來的臉 放在 faces 中
  faces = cvHaarDetectObjects( tempFrame, classifier, facesMemStorage, 1.1, 3, 
    CV_HAAR_DO_CANNY_PRUNING, cvSize(min_face_width, min_face_height ) );

  namedWindow("Face Detection Result", 1); // 建立一個視窗 命名為 Face Detection Result
  CvFont font;
  double hScale = 1.0;
  double vScale = 1.0;
  int    lineWidth = 2;
  cvInitFont( &font, CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale, vScale, 0, lineWidth );

  // 將所有的 face 以紅線框起來
  if ( faces ) {
    for ( int i = 0 ; i < faces->total ; ++i ) {
      // Setup two points that define the extremes of the rectangle,
      // then draw it to the image
      CvPoint point1, point2;
      CvRect* rectangle = (CvRect*)cvGetSeqElem(faces, i);
      point1.x = rectangle->x;
      point2.x = rectangle->x + rectangle->width;
      point1.y = rectangle->y;
      point2.y = rectangle->y + rectangle->height;
      cvRectangle( tempFrame, point1, point2, CV_RGB(255,0,0), 3, 8, 0);
      stringstream temp ;
      temp << (char)('A' + i) ;
      string w ;
      temp >> w ;
      cvPutText( tempFrame, w.c_str(), Point( point1.x + 2, point1.y + 25 ),  &font, cvScalar( 200, 200, 100 ) ) ;
      temp.clear() ;
    } // for
  } // if

  imshow("Face Detection Result", Mat( tempFrame ) ) ; // 將 tempFrame show 在視窗中
} // ShowDetectFace()

bool ChangeFace( CvSeq *faces, Img &mainImg, int faceNum, int headIndex ) { 
  // 將使用者所輸入的 faceNum 與 headIndex 
  // 對應到哪一張臉 與 哪一個動物 進行 把 動物圖 放到 臉上 的動作
  if ( !faces ) {
    cout << "error : no face" << endl ;
    return false ;
  } // if

  if ( faces->total <= faceNum || faceNum < 0 ) {
    cout << "error : faceNum out of range" << endl ;
    return false ;
  } // if

  if ( headIndex >= ANI_HEAD_MAX || headIndex < 0 ) {
    cout << "error : headIndex out of range" << endl ;
    return false ;
  } // if

  CvPoint point1;
  CvRect* rectangle = (CvRect*)cvGetSeqElem( faces, faceNum ); // 取得使用者要的是哪一張臉
  // 將 臉的座標 抓出來
  point1.x = rectangle->x;
  point1.y = rectangle->y;
  Mat reScaleTemp;
  Mat imgROI ;
  // 將 動物圖 以 臉的框框 來重新縮放寬高
  resize( gAniHead[ headIndex ].img, reScaleTemp, Size( rectangle->width, rectangle->height ) ) ; 
  imgROI = mainImg.img( Rect( point1.x, point1.y, reScaleTemp.cols, reScaleTemp.rows ) );

  if ( !cvAdd4cMat_q( imgROI, reScaleTemp, 1.0 ) ) // 將 動物圖 與原圖融合
    addWeighted(imgROI, 0, reScaleTemp, 1.0, 0, imgROI );
  
  imshow("Change Face", mainImg.img ) ; // 將融合好的圖放到視窗中

} // ChangeFace()

int main( int argc , char ** argv ){

  /* init head */
  /* 載入所有動物圖 */
  for ( int i = 0 ; i < ANI_HEAD_MAX ; i++ ) {
    gAniHead[i] = Img( gHeadName[i], "./animalHead/" ) ;
  } // for

  ShowAnimalHead() ; // 將載入好的動物圖 show 在視窗上

  Img mainImg( "XD2.jpg" ) ; // 載入你想偵測的臉
  CvSeq *faces = NULL ;
  ShowDetectFace( mainImg, faces ) ; // 將臉偵測完畢 畫框show在視窗上

  namedWindow("Change Face", 1); // 開啟一個視窗 名叫 "Change Face"

  string faceNum = "init" ;
  int head = 0 ;

  while ( faceNum.at( 0 ) != '-' ) {
    cvWaitKey( 5000 ) ; // 等候 5 秒 讓所有視窗繪圖完成
    cin >> faceNum >> head ; // 讀取使用者輸入 臉的編號:字母  動物圖的編號:數字
    if ( faceNum.at( 0 ) == '-' )
      break ;
    ChangeFace( faces, mainImg, faceNum.at( 0 ) - 'A', head ) ; // 將使用者輸入之字母換成數字 傳入將動物圖放到臉上
  } // while

  cout << "press ctrl + c or click photo and press any key to quit" << endl ; 
  cvWaitKey( 0 );
  return EXIT_SUCCESS;
}
