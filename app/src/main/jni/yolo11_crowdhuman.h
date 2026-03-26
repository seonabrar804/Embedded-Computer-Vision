// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef YOLO11_CROWDMAN_H
#define YOLO11_CROWDMAN_H

#include "yolo11.h"

class YOLO11_crowdhuman : public YOLO11
{
public:
    virtual int detect(const cv::Mat& rgb, std::vector<Object>& objects);
    virtual int draw(cv::Mat& rgb, const std::vector<Object>& objects);
};

#endif // YOLO11_CROWDMAN_H