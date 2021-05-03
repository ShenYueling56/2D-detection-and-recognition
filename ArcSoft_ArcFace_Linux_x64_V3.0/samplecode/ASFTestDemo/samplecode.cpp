#include "arcsoft_face_sdk.h"
#include "amcomdef.h"
#include "asvloffscreen.h"
#include "merror.h"
#include <iostream>  
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include<opencv2/opencv.hpp>
#include <fstream>
#include <opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;


//从开发者中心获取APPID/SDKKEY(以下均为假数据，请替换)
#define APPID "8syZbU7nFoG4FsP6uegGMZqZTvieuSCohh8qeS1nXkEZ"
#define SDKKEY "6UsUNBGuPgVVRcXeACsEtBHi8teHCs3MmJirSqmqjFaR"

#define NSCALE 16 
#define FACENUM	5

#define SafeFree(p) { if ((p)) free(p); (p) = NULL; }
#define SafeArrayDelete(p) { if ((p)) delete [] (p); (p) = NULL; } 
#define SafeDelete(p) { if ((p)) delete (p); (p) = NULL; } 

//时间戳转换为日期格式
void timestampToTime(char* timeStamp, char* dateTime, int dateTimeSize)
{
	time_t tTimeStamp = atoll(timeStamp);
	struct tm* pTm = gmtime(&tTimeStamp);
	strftime(dateTime, dateTimeSize, "%Y-%m-%d %H:%M:%S", pTm);
}

//图像颜色格式转换
int ColorSpaceConversion(MInt32 width, MInt32 height, MInt32 format, MUInt8* imgData, ASVLOFFSCREEN& offscreen)
{
	offscreen.u32PixelArrayFormat = (unsigned int)format;
	offscreen.i32Width = width;
	offscreen.i32Height = height;
	
	switch (offscreen.u32PixelArrayFormat)
	{
	case ASVL_PAF_RGB24_B8G8R8:
		offscreen.pi32Pitch[0] = offscreen.i32Width * 3;
		offscreen.ppu8Plane[0] = imgData;
		break;
	case ASVL_PAF_I420:
		offscreen.pi32Pitch[0] = width;
		offscreen.pi32Pitch[1] = width >> 1;
		offscreen.pi32Pitch[2] = width >> 1;
		offscreen.ppu8Plane[0] = imgData;
		offscreen.ppu8Plane[1] = offscreen.ppu8Plane[0] + offscreen.i32Height*offscreen.i32Width;
		offscreen.ppu8Plane[2] = offscreen.ppu8Plane[0] + offscreen.i32Height*offscreen.i32Width * 5 / 4;
		break;
	case ASVL_PAF_NV12:
	case ASVL_PAF_NV21:
		offscreen.pi32Pitch[0] = offscreen.i32Width;
		offscreen.pi32Pitch[1] = offscreen.pi32Pitch[0];
		offscreen.ppu8Plane[0] = imgData;
		offscreen.ppu8Plane[1] = offscreen.ppu8Plane[0] + offscreen.pi32Pitch[0] * offscreen.i32Height;
		break;
	case ASVL_PAF_YUYV:
	case ASVL_PAF_DEPTH_U16:
		offscreen.pi32Pitch[0] = offscreen.i32Width * 2;
		offscreen.ppu8Plane[0] = imgData;
		break;
	case ASVL_PAF_GRAY:
		offscreen.pi32Pitch[0] = offscreen.i32Width;
		offscreen.ppu8Plane[0] = imgData;
		break;
	default:
		return 0;
	}
	return 1;
}

struct Target
{
    int ID=0;
    char* name;
    int total_num=-1;
    MRECT faceRect;
    ASF_FaceFeature feature;
    bool is_painted = false;
};
//opencv方式裁剪图片
void CutIplImage(IplImage *src, IplImage *dst, int x, int y)
{
    CvSize size = cvSize(dst->width, dst->height);//区域大小
    cvSetImageROI(src, cvRect(x, y, size.width, size.height));//设置源图像ROI
    cvCopy(src, dst); //复制图像
    cvResetImageROI(src);//源图像用完后,清空ROI
}

void ASFFaceFaceTargetGet(MHandle handle,MRESULT res,char* img_path,Target *target)
{
    IplImage*  originalImg = cvLoadImage(img_path,CV_LOAD_IMAGE_COLOR);
    //图像裁剪,宽度做四字节对齐,若能保证图像是四字节对齐这步可以不用做
    IplImage* img = cvCreateImage(cvSize(originalImg->width - originalImg->width % 4, originalImg->height), IPL_DEPTH_8U, originalImg->nChannels);
    CutIplImage(originalImg, img, 0, 0);
    //图像数据以结构体形式传入,对更高精度的图像兼容性更好
    ASVLOFFSCREEN offscreen = { 0 };
    offscreen.u32PixelArrayFormat = ASVL_PAF_RGB24_B8G8R8;
    offscreen.i32Width = img->width;
    offscreen.i32Height = img->height;
    offscreen.pi32Pitch[0] = img->widthStep;
    offscreen.ppu8Plane[0] = (MUInt8*)img->imageData;
    ASF_MultiFaceInfo detectedFaces = { 0 };

    if (img) {
        res = ASFDetectFacesEx(handle, &offscreen, &detectedFaces);
        if (MOK != res)
        {
            printf("ASFDetectFacesEx failed: %d\n", res);
        }
        else
        {
            // 打印人脸检测结果
            for (int i = 0; i < detectedFaces.faceNum; i++)
            {//printf("Face Id: %d\n", detectedFaces4.faceID[i]);
                printf("Face Orient: %d\n", detectedFaces.faceOrient[i]);
                printf("Face Rect: (%d %d %d %d)\n",
                       detectedFaces.faceRect[i].left, detectedFaces.faceRect[i].top,
                       detectedFaces.faceRect[i].right,
                       detectedFaces.faceRect[i].bottom);
                target[i].faceRect.left=detectedFaces.faceRect[i].left;
                target[i].faceRect.right=detectedFaces.faceRect[i].right;
                target[i].faceRect.top=detectedFaces.faceRect[i].top;
                target[i].faceRect.bottom=detectedFaces.faceRect[i].bottom;
                target[i].total_num = detectedFaces.faceNum;
            }
        }
        //可以对人脸大小进行排序,取最大人脸检测,可根据实际应用场景进行选择。
        for (int i = 0; i < target[i].total_num; i++) {
            ASF_FaceFeature feature = {0};
            ASF_SingleFaceInfo singleDetectedFaces = {0};
            singleDetectedFaces.faceRect.left = detectedFaces.faceRect[i].left;
            singleDetectedFaces.faceRect.top = detectedFaces.faceRect[i].top;
            singleDetectedFaces.faceRect.right = detectedFaces.faceRect[i].right;
            singleDetectedFaces.faceRect.bottom = detectedFaces.faceRect[i].bottom;
            singleDetectedFaces.faceOrient = detectedFaces.faceOrient[i];
            res = ASFFaceFeatureExtract(handle, img->width, img->height,
                                        ASVL_PAF_RGB24_B8G8R8, (MUInt8 *) img->imageData,
                                        &singleDetectedFaces, &feature);

            if (MOK != res) {
                printf("ASFFaceFeatureExtract failed: %d\n", res);
                //也拷贝feature
                target[i].feature.featureSize = feature.featureSize;
                target[i].feature.feature = (MByte *) malloc(feature.featureSize);
                memset(target[i].feature.feature, 0, feature.featureSize);
                memcpy(target[i].feature.feature, feature.feature, feature.featureSize);
            }else {
                //拷贝feature，否则第二次进行特征提取，会覆盖第一次特征提取的数据，导致比对的结果为1

                target[i].feature.featureSize = feature.featureSize;
                target[i].feature.feature = (MByte *) malloc(feature.featureSize);
                memset(target[i].feature.feature, 0, feature.featureSize);
                memcpy(target[i].feature.feature, feature.feature, feature.featureSize);
            }
        }
    }



}

static Scalar randomColor( RNG& rng )
{
    int icolor = (unsigned) rng;
    return Scalar( icolor&255, (icolor>>8)&255, (icolor>>16)&255 );
}

void ASFFaceVisualization(MHandle handle,MRESULT res,char* img_path1,char* img_path2,
                          Target *target1,Target *target2)
{

    Mat src1 =imread(img_path1);
    Mat src2 =imread(img_path2);
    MFloat confidenceLevel;
    RNG rng( 0xFFFFFFFF );
    for (int i = 0; i < target1[i].total_num; i++) {
        int x = target1[i].faceRect.left;
        int y = target1[i].faceRect.top;
        int width = target1[i].faceRect.right-target1[i].faceRect.left;
        int height = -target1[i].faceRect.top + target1[i].faceRect.bottom;
        Scalar scalar = randomColor(rng);
        rectangle(src1,cvRect(x,y,width,height),scalar,5);
        target1[i].is_painted = true;
        for (int j = 0; j < target2[j].total_num; j++) {
            res = ASFFaceFeatureCompare(handle, &target1[i].feature, &target2[j].feature, &confidenceLevel);
            if (confidenceLevel>0.8) {
                printf("ASFFaceFeatureCompare sucess: %lf\n", confidenceLevel);
                int x = target2[j].faceRect.left;
                int y = target2[j].faceRect.top;
                int width = target2[j].faceRect.right - target2[j].faceRect.left;
                int height = -target2[j].faceRect.top + target2[j].faceRect.bottom;
                rectangle(src2, cvRect(x, y, width, height), scalar, 5);
                target2[j].is_painted = true;
            }
        }
    }
    for (int j = 0; j < target2[j].total_num; j++) {
        if(target2[j].is_painted)
            continue;
        int x = target2[j].faceRect.left;
        int y = target2[j].faceRect.top;
        int width = target2[j].faceRect.right-target2[j].faceRect.left;
        int height = -target2[j].faceRect.top + target2[j].faceRect.bottom;
        rectangle(src2,cvRect(x,y,width,height),randomColor(rng),5);
        target2[j].is_painted = true;
    }


   // resize(src2, src2, Size(1200, 1000));
    imshow("2",src2);
    waitKey(0);
    imwrite("./result.jpg",src2);
    Mat two;
    int rows = 600;
    int cols = 500;
    resize(src1, src1, Size(cols, rows));
    resize(src2, src2, Size(cols, rows));
    hconcat(src1,src2,two);
    imshow("two images",two);
    waitKey(0);
}



int main() {

    srand((unsigned)time(NULL));
    printf("\n************* ArcFace SDK Info *****************\n");
    MRESULT res = MOK;
    ASF_ActiveFileInfo activeFileInfo = {0};
    res = ASFGetActiveFileInfo(&activeFileInfo);
    if (res != MOK) {
        printf("ASFGetActiveFileInfo fail: %d\n", res);
    } else {
        //这里仅获取了有效期时间，还需要其他信息直接打印即可
        char startDateTime[32];
        timestampToTime(activeFileInfo.startTime, startDateTime, 32);
        printf("startTime: %s\n", startDateTime);
        char endDateTime[32];
        timestampToTime(activeFileInfo.endTime, endDateTime, 32);
        printf("endTime: %s\n", endDateTime);
    }

    //SDK版本信息
    const ASF_VERSION version = ASFGetVersion();
    printf("\nVersion:%s\n", version.Version);
    printf("BuildDate:%s\n", version.BuildDate);
    printf("CopyRight:%s\n", version.CopyRight);

    printf("\n************* Face Recognition *****************\n");

    res = ASFOnlineActivation(APPID, SDKKEY);
    if (MOK != res && MERR_ASF_ALREADY_ACTIVATED != res)
        printf("ASFOnlineActivation fail: %d\n", res);
    else
        printf("ASFOnlineActivation sucess: %d\n", res);

    //初始化引擎
    MHandle handle = NULL;
    MInt32 nScale = 2;
    MInt32 faceNum = 100;
    MInt32 mask = ASF_FACE_DETECT | ASF_FACERECOGNITION | ASF_AGE | ASF_GENDER |
                  ASF_FACE3DANGLE | ASF_LIVENESS | ASF_IR_LIVENESS;
    res = ASFInitEngine(ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, nScale,
                        faceNum, mask, &handle);
    if (res != MOK)
        printf("ASFInitEngine fail: %d\n", res);
    else
        printf("ASFInitEngine sucess: %d\n", res);

	char* img_path1="../images/two_people.jpg";
    char* img_path2="../images/test.jpg";
    Target target1[100];
    Target target2[1000];

    ASFFaceFaceTargetGet(handle,res,img_path1,target1);
    ASFFaceFaceTargetGet(handle,res,img_path2,target2);

    ASFFaceVisualization(handle,res,img_path1,img_path2,target1,target2);





    /*********以下三张图片均存在，图片保存在 ./bulid/images/ 文件夹下*********/

	//可见光图像 NV21格式裸数据
	/*
	char* picPath1 = "../images/640x480_1.NV21";
	int Width1 = 640;
	int Height1 = 480;
	int Format1 = ASVL_PAF_NV21;
	MUInt8* imageData1 = (MUInt8*)malloc(Height1*Width1*3/2);
	FILE* fp1 = fopen(picPath1, "rb");
	
	//可见光图像 NV21格式裸数据
	char* picPath2 = "../images/640x480_2.NV21";
	int Width2 = 640;
	int Height2 = 480;
	int Format2 = ASVL_PAF_NV21;
	MUInt8* imageData2 = (MUInt8*)malloc(Height1*Width1*3/2);
	FILE* fp2 = fopen(picPath2, "rb");
	
	//红外图像 NV21格式裸数据
	char* picPath3 = "../images/640x480_3.NV21";
	int Width3 = 640;
	int Height3 = 480;
	int Format3 = ASVL_PAF_GRAY;
	MUInt8* imageData3 = (MUInt8*)malloc(Height2*Width2);	//只读NV21前2/3的数据为灰度数据
	FILE* fp3 = fopen(picPath3, "rb");

	if (fp1 && fp2 && fp3)
	{
		fread(imageData1, 1, Height1*Width1*3/2, fp1);	//读NV21裸数据
		fclose(fp1);
		fread(imageData2, 1, Height1*Width1*3/2, fp2);	//读NV21裸数据
		fclose(fp2);
		fread(imageData3, 1, Height3*Width3, fp3);		//读NV21前2/3的数据,用于红外活体检测
		fclose(fp3);

		ASVLOFFSCREEN offscreen1 = { 0 };
		ColorSpaceConversion(Width1, Height1, ASVL_PAF_NV21, imageData1, offscreen1);
		
		//第一张人脸
		ASF_MultiFaceInfo detectedFaces1 = { 0 };
		ASF_SingleFaceInfo SingleDetectedFaces = { 0 };
		ASF_FaceFeature feature1 = { 0 };
		ASF_FaceFeature copyfeature1 = { 0 };
		
		res = ASFDetectFacesEx(handle, &offscreen1, &detectedFaces1);;
		if (res != MOK && detectedFaces1.faceNum > 0)
		{
			printf("%s ASFDetectFaces 1 fail: %d\n", picPath1, res);
		}
		else
		{
			SingleDetectedFaces.faceRect.left = detectedFaces1.faceRect[0].left;
			SingleDetectedFaces.faceRect.top = detectedFaces1.faceRect[0].top;
			SingleDetectedFaces.faceRect.right = detectedFaces1.faceRect[0].right;
			SingleDetectedFaces.faceRect.bottom = detectedFaces1.faceRect[0].bottom;
			SingleDetectedFaces.faceOrient = detectedFaces1.faceOrient[0];
			
			// 单人脸特征提取
			res = ASFFaceFeatureExtractEx(handle, &offscreen1, &SingleDetectedFaces, &feature1);
			if (res != MOK)
			{
				printf("%s ASFFaceFeatureExtractEx 1 fail: %d\n", picPath1, res);
			}
			else
			{
				//拷贝feature，否则第二次进行特征提取，会覆盖第一次特征提取的数据，导致比对的结果为1
				copyfeature1.featureSize = feature1.featureSize;
				copyfeature1.feature = (MByte *)malloc(feature1.featureSize);
				memset(copyfeature1.feature, 0, feature1.featureSize);
				memcpy(copyfeature1.feature, feature1.feature, feature1.featureSize);
			}
		}
		
		//第二张人脸
		ASVLOFFSCREEN offscreen2 = { 0 };
		ColorSpaceConversion(Width2, Height2, ASVL_PAF_NV21, imageData2, offscreen2);
		
		ASF_MultiFaceInfo detectedFaces2 = { 0 };
		ASF_FaceFeature feature2 = { 0 };
		
		res = ASFDetectFacesEx(handle, &offscreen2, &detectedFaces2);
		if (res != MOK && detectedFaces2.faceNum > 0)
		{
			printf("%s ASFDetectFacesEx 2 fail: %d\n", picPath2, res);
		}
		else
		{
			SingleDetectedFaces.faceRect.left = detectedFaces2.faceRect[0].left;
			SingleDetectedFaces.faceRect.top = detectedFaces2.faceRect[0].top;
			SingleDetectedFaces.faceRect.right = detectedFaces2.faceRect[0].right;
			SingleDetectedFaces.faceRect.bottom = detectedFaces2.faceRect[0].bottom;
			SingleDetectedFaces.faceOrient = detectedFaces2.faceOrient[0];
			
			res = ASFFaceFeatureExtractEx(handle, &offscreen2, &SingleDetectedFaces, &feature2);
			if (res != MOK)
				printf("%s ASFFaceFeatureExtractEx 2 fail: %d\n", picPath2, res);
			else
				printf("%s ASFFaceFeatureExtractEx 2 sucess: %d\n", picPath2, res);
		}

		// 单人脸特征比对
		MFloat confidenceLevel;
		printf("...................................\n");
		//res = ASFFaceFeatureCompare(handle, &copyfeature1, &feature2, &confidenceLevel);
        res = ASFFaceFeatureCompare(handle, &feature2, &feature, &confidenceLevel);

		if (res != MOK)
			printf("ASFFaceFeatureCompare fail: %d\n", res);
		else
			printf("ASFFaceFeatureCompare sucess: %lf\n", confidenceLevel);
		

		printf("\n************* Face Process *****************\n");
		//设置活体置信度 SDK内部默认值为 IR：0.7  RGB：0.5（无特殊需要，可以不设置）
		ASF_LivenessThreshold threshold = { 0 };
		threshold.thresholdmodel_BGR = 0.5;
		threshold.thresholdmodel_IR = 0.7;
		res = ASFSetLivenessParam(handle, &threshold);
		if (res != MOK)
			printf("ASFSetLivenessParam fail: %d\n", res);
		else
			printf("RGB Threshold: %f\nIR Threshold: %f\n", threshold.thresholdmodel_BGR, threshold.thresholdmodel_IR);

		// 人脸信息检测
		MInt32 processMask = ASF_AGE | ASF_GENDER | ASF_FACE3DANGLE | ASF_LIVENESS;
		res = ASFProcessEx(handle, &offscreen2, &detectedFaces2, processMask);
		if (res != MOK)
			printf("ASFProcessEx fail: %d\n", res);
		else
			printf("ASFProcessEx sucess: %d\n", res);

		// 获取年龄
		ASF_AgeInfo ageInfo = { 0 };
		res = ASFGetAge(handle, &ageInfo);
		if (res != MOK)
			printf("%s ASFGetAge fail: %d\n", picPath2, res);
		else
			printf("%s First face age: %d\n", picPath2, ageInfo.ageArray[0]);

		// 获取性别
		ASF_GenderInfo genderInfo = { 0 };
		res = ASFGetGender(handle, &genderInfo);
		if (res != MOK)
			printf("%s ASFGetGender fail: %d\n", picPath2, res);
		else
			printf("%s First face gender: %d\n", picPath2, genderInfo.genderArray[0]);

		// 获取3D角度
		ASF_Face3DAngle angleInfo = { 0 };
		res = ASFGetFace3DAngle(handle, &angleInfo);
		if (res != MOK)
			printf("%s ASFGetFace3DAngle fail: %d\n", picPath2, res);
		else
			printf("%s First face 3dAngle: roll: %lf yaw: %lf pitch: %lf\n", picPath2, angleInfo.roll[0], angleInfo.yaw[0], angleInfo.pitch[0]);
		
		//获取活体信息
		ASF_LivenessInfo rgbLivenessInfo = { 0 };
		res = ASFGetLivenessScore(handle, &rgbLivenessInfo);
		if (res != MOK)
			printf("ASFGetLivenessScore fail: %d\n", res);
		else
			printf("ASFGetLivenessScore sucess: %d\n", rgbLivenessInfo.isLive[0]);
		
		
		printf("\n**********IR LIVENESS*************\n");
		
		//第二张人脸
		ASVLOFFSCREEN offscreen3 = { 0 };
		ColorSpaceConversion(Width3, Height3, ASVL_PAF_GRAY, imageData3, offscreen3);
		
		ASF_MultiFaceInfo detectedFaces3 = { 0 };
		res = ASFDetectFacesEx(handle, &offscreen3, &detectedFaces3);
		if (res != MOK)
			printf("ASFDetectFacesEx fail: %d\n", res);
		else
			printf("Face num: %d\n", detectedFaces3.faceNum);
		
		//IR图像活体检测
		MInt32 processIRMask = ASF_IR_LIVENESS;
		res = ASFProcessEx_IR(handle, &offscreen3, &detectedFaces3, processIRMask);
		if (res != MOK)
			printf("ASFProcessEx_IR fail: %d\n", res);
		else
			printf("ASFProcessEx_IR sucess: %d\n", res);
		
		//获取IR活体信息
		ASF_LivenessInfo irLivenessInfo = { 0 };
		res = ASFGetLivenessScore_IR(handle, &irLivenessInfo);
		if (res != MOK)
			printf("ASFGetLivenessScore_IR fail: %d\n", res);
		else
			printf("IR Liveness: %d\n", irLivenessInfo.isLive[0]);
		
		//释放内存
		SafeFree(copyfeature1.feature);
		SafeArrayDelete(imageData1);
		SafeArrayDelete(imageData2);
		SafeArrayDelete(imageData3);
		

		//反初始化
		res = ASFUninitEngine(handle);
		if (res != MOK)
			printf("ASFUninitEngine fail: %d\n", res);
		else
			printf("ASFUninitEngine sucess: %d\n", res);
	}
	else
	{
		printf("No pictures found.\n");
	}

	getchar();
 */
    //释放内存???
    //反初始化
    res = ASFUninitEngine(handle);
    if (res != MOK)
        printf("ASFUninitEngine fail: %d\n", res);
    else
        printf("ASFUninitEngine sucess: %d\n", res);
    return 0;
}

