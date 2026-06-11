/*
 * This file is part of the JuMarker library
 * Copyright (c) 2022 David Jurado Rodríguez, Rafael Muñoz Salinas,
 *                    Sergio Garrido Jurado, Rafael Medina Carnicer.
 *
 * JuMarker is published and distributed under the CC BY-NC-SA license.
 * JuMarker is distributed in the hope that it will be useful for
 * non-commercial academic research, but WITHOUT ANY WARRANTY.
 *
 * You should have received a copy of the CC BY-NC-SA license along with
 * this program; if not, write to rmsalinas@uco.es.
 */

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
inline cv::Point2f __projectPoint(const cv::Point3f &objectPoint, const double *R /*3x3 Rotation Matrix (cv::Rodrigues)*/,
                                  const double *tvec /*3*/, const double fx, const double fy, const double cx, const double cy,
                                  const double *dist /*5*/
)
{

    double k[14] = {dist[0], dist[1], dist[2], dist[3], dist[4], 0, 0, 0, 0, 0, 0, 0, 0, 0};
    cv::Matx33d matTilt = cv::Matx33d::eye();
    cv::Point2d imagePoint;
    cv::Vec3d vecTilt;

    double X = objectPoint.x, Y = objectPoint.y, Z = objectPoint.z;
    double x = R[0] * X + R[1] * Y + R[2] * Z + tvec[0];
    double y = R[3] * X + R[4] * Y + R[5] * Z + tvec[1];
    double z = R[6] * X + R[7] * Y + R[8] * Z + tvec[2];
    double r2, r4, r6, a1, a2, a3, cdist, icdist2;
    double xd, yd, xd0, yd0, invProj;

    z = z ? 1. / z : 1;
    x *= z;
    y *= z;

    r2 = x * x + y * y;
    r4 = r2 * r2;
    r6 = r4 * r2;
    a1 = 2 * x * y;
    a2 = r2 + 2 * x * x;
    a3 = r2 + 2 * y * y;
    cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6;
    icdist2 = 1 / (1 + k[5] * r2 + k[6] * r4 + k[7] * r6);
    xd0 = x * cdist * icdist2 + k[2] * a1 + k[3] * a2 + k[8] * r2 + k[9] * r4;
    yd0 = y * cdist * icdist2 + k[2] * a3 + k[3] * a1 + k[10] * r2 + k[11] * r4;

    // additional distortion by projecting onto a tilt plane
    vecTilt = matTilt * cv::Vec3d(xd0, yd0, 1);
    invProj = vecTilt(2) ? 1 / vecTilt(2) : 1;
    xd = invProj * vecTilt(0);
    yd = invProj * vecTilt(1);

    imagePoint.x = xd * fx + cx;
    imagePoint.y = yd * fy + cy;

    return imagePoint;
}
