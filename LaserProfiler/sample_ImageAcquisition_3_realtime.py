# -*- coding: 'unicode' -*-
# Copyright (c) 2021 KEYENCE CORPORATION. All rights reserved.

import LJXAwrap

import ctypes
import sys
import time
import pandas as pd


profileData = []



def main():

    deviceId = 0  # Set "0" if you use only 1 head.
    ethernetConfig = LJXAwrap.LJX8IF_ETHERNET_CONFIG()
    ethernetConfig.abyIpAddress[0] = 192  # IP address
    ethernetConfig.abyIpAddress[1] = 168
    ethernetConfig.abyIpAddress[2] = 0
    ethernetConfig.abyIpAddress[3] = 1
    ethernetConfig.wPortNo = 24691        # Port No.

    res = LJXAwrap.LJX8IF_EthernetOpen(deviceId, ethernetConfig)
    print("LJXAwrap.LJX8IF_EthernetOpen:", hex(res))
    if res != 0:
        print("Failed to connect contoller.")
        sys.exit()
    print("----0")

    ##################################################################
    # sample_HowToCallFunctions.py:
    #  A sample collection of how to call LJXAwrap I/F functions.
    #
    # Conditional branch of each sample is initially set to 'False'.
    # This is to prevent accidental execution. Set 'True' to execute.
    #
    # <NOTE> Controller settings may change in some sample codes.
    #
    ##################################################################
    if True:
        # Example of how to get profile data.
        #
        # <NOTE>
        # -This method is suitable for reading a few profile data.
        #
        # -Use high-speed communication method to acquire a large amount
        #  of profiles, such as height or luminance image data.
        #  For details, refer to another sample (sample_ImageAcquisition.py)

        # Change according to your controller settings.
        xpointNum = 3200            # Number of X points per one profile.
        withLumi = 1                # 1: luminance data exists, 0: not exists.

        # Specifies the position, etc. of the profiles to get.
        req = LJXAwrap.LJX8IF_GET_PROFILE_REQUEST()
        req.byTargetBank = 0x0      # 0: active bank
        req.byPositionMode = 0x0    # 0: from current position
        req.dwGetProfileNo = 0x0    # use when position mode is "POSITION_SPEC"
        req.byGetProfileCount = 1   # the number of profiles to read.
        req.byErase = 0             # 0: Do not erase

        rsp = LJXAwrap.LJX8IF_GET_PROFILE_RESPONSE()

        profinfo = LJXAwrap.LJX8IF_PROFILE_INFO()

        # Calculate the buffer size to store the received profile data.
        dataSize = ctypes.sizeof(LJXAwrap.LJX8IF_PROFILE_HEADER)
        dataSize += ctypes.sizeof(LJXAwrap.LJX8IF_PROFILE_FOOTER)
        dataSize += ctypes.sizeof(ctypes.c_uint) * xpointNum * (1 + withLumi)
        dataSize *= req.byGetProfileCount

        dataNumIn4byte = int(dataSize / ctypes.sizeof(ctypes.c_uint))
        profdata = (ctypes.c_int * dataNumIn4byte)()

        # Send command.
        res = LJXAwrap.LJX8IF_GetProfile(deviceId,
                                         req,
                                         rsp,
                                         profinfo,
                                         profdata,
                                         dataSize)

        print("LJXAwrap.LJX8IF_GetProfile:", hex(res))
        if res != 0:
            print("Failed to get profile.")
            sys.exit()

        print("----------------------------------------")
        print(" byLuminanceOutput     :", profinfo.byLuminanceOutput)
        print(" wProfileDataCount(X)  :", profinfo.wProfileDataCount)
        print(" lXPitch(in 0.01um)    :", profinfo.lXPitch)
        print(" lXStart(in 0.01um)    :", profinfo.lXStart)
        print("-----")
        print(" dwCurrentProfileNo    :", rsp.dwCurrentProfileNo)
        print(" dwOldestProfileNo     :", rsp.dwOldestProfileNo)
        print(" dwGetTopProfileNo     :", rsp.dwGetTopProfileNo)
        print(" byGetProfileCount     :", rsp.byGetProfileCount)
        print("----------------------------------------")

        headerSize = ctypes.sizeof(LJXAwrap.LJX8IF_PROFILE_HEADER)
        addressOffset_height = int(headerSize / ctypes.sizeof(ctypes.c_uint))
        addressOffset_lumi = addressOffset_height + profinfo.wProfileDataCount

        for i in range(profinfo.wProfileDataCount):
            # Conver X data to the actual length in millimeters
            x_val_mm = (profinfo.lXStart + profinfo.lXPitch * i) / 100.0  # um
            x_val_mm /= 1000.0  # mm

            # Conver Z data to the actual length in millimeters
            z_val = profdata[addressOffset_height + i]

            if z_val <= -2147483645:  # invalid value
                z_val_mm = - 999.9999
            else:
                z_val_mm = z_val / 100.0  # um
                z_val_mm /= 1000.0  # mm

            # Luminance data
            lumi_val = profdata[addressOffset_lumi + i]

            print('{:.04f}'.format(x_val_mm),
                  '{:.04f}'.format(z_val_mm),
                  lumi_val)


            profileData.append({
                        'X': x_val_mm,
                        'Z': z_val_mm
            })

        profile_df = pd.DataFrame(profileData)
        print(profile_df)
        # Save the DataFrame to a CSV file
        profile_df.to_csv('sample_ImageAcquisition.csv', index=False)







            # # saving the data into CSV file
            # import pandas as pd
            # df_x_val_mm = pd.DataFrame(x_val_mm)
            # df_z_val_mm = pd.DataFrame(z_val_mm)
            #
            # df_new = pd.DataFrame(
            #     {
            #         'x val': x_val_mm,
            #         'z val': z_val_mm
            #     }
            # )
            #
            # # Save the DataFrame to a CSV file
            # df_x_val_mm.to_csv('sample_ImageAcquisition_3_realtime.csv', index=False)

        print("----")

    res = LJXAwrap.LJX8IF_CommunicationClose(deviceId)
    print("LJXAwrap.LJX8IF_CommunicationClose:", hex(res))

    return


if __name__ == '__main__':
    main()
