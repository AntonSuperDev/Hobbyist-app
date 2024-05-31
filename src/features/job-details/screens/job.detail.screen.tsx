/* eslint-disable prettier/prettier */
/* eslint-disable react-native/no-inline-styles */
/* eslint-disable react-hooks/exhaustive-deps */

import {Alert, Dimensions, Linking, Text, TextInput, View} from 'react-native';
import {
  BackgroundBottom,
  BackgroundTop,
} from '../../../components/backgrounds/backgroundImage';
import {
  ChartContainer,
  ItemAddedDateText,
  ItemDescription,
  ItemDescriptionDetail,
  ItemDescriptionDetailHeader,
  ItemDescriptionEditImage,
  ItemDescriptionImage,
  ItemDescriptionText,
  ItemDetailContainer,
  ItemDetailContainerPadding,
  ItemEditImage,
  ItemGoBack,
  ItemGoBackImage,
  ItemTier,
  ItemTierDivider,
  ItemTitleContainer,
  ItemUserTier,
  ItemUserTierImage,
  ItemUserTierTitleText,
  ItemUserTierVariant,
  OnScroll,
  OnScrollHorizontal,
  OnTouch,
  OnTouchEditImage,
  ScrollViwWrapper,
  TierImage,
  TierLevelIcon,
  TierLevelText,
  UserTierText,
} from '../components/job.detail.styles';
import React, {FC, useCallback, useEffect, useState} from 'react';

import AsyncStorage from '@react-native-async-storage/async-storage';
import Chart from '../components/charts';
import CheckIcon from '../../../../assets/check-green.png';
import Edit from '../../../../assets/edit.png';
import EditIcon from '../../../../assets/edit-icon.png';
import EditImage from '../../../../assets/edit-image.png';
import HeaderContainer from '../../../components/header-components/header.component';
import ImagePicker from 'react-native-image-crop-picker';
import Moment from 'moment';
import {NoItem} from '../../Home/components/home.styles';
import {SafeView} from '../../../components/utility/safeAreaView.component';
import SampleImage from '../../../../assets/sample-image.png';
import {Spacer} from '../../../components/utility/spacer';
import UserTier from '../../../../assets/user-tier.png';
import {Wrapper} from '../../../components/utility/wrapperContainer';
import X from '../../../../assets/x.png';
import { BASE_URL } from '../../../config';

// import {configureStore} from '@reduxjs/toolkit';

/* eslint-disable @typescript-eslint/no-unused-vars */
interface Props {
  navigation: any;
  route: any;
}
export const getWidth = Dimensions.get('window').width;
const JobDetailScreen: FC<Props> = ({navigation, route}) => {
  const {item} = route.params;
  const [isEditDesc, setIsEditDesc] = useState(false);
  const [isEditTitle, setIsEditTitle] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [editDesc, setEditDesc] = useState('');
  const [editTitle, setEditTitle] = useState('');
  const [imageP, setImageP] = useState<string | null>(null);
  const [similarItems, setSimilaritems] = useState([]);
  const [sameData, setSameData] = useState([]);
  const [dataItem, setDataItem] = useState([]);
  const [dataCat, setDataCat] = useState([]);
  const [imageUrl, setImageUrl] = useState(null);
  const [imageId, setImageId] = useState(null);
  const [user, setUser] = useState('tier-I');

  // const dispatch = useDispatch();
  const handleClick = (url: string) => {
    Linking.openURL(url);
    // Linking.canOpenURL(url).then(supported => {
    //   if (supported) {
    //     Linking.openURL('https://google.com');
    //   } else {
    //     console.log("Don't know how to open URI: " + url);
    //   }
    // });
  };
  const getSimilarItems = async () => {
    const token = await AsyncStorage.getItem('token');
    var apiUrl = `${BASE_URL}/api/v1/get-similar-item/${item._id}`;

    var headers = {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + token,
    };
    fetch(apiUrl, {
      method: 'GET',
      headers: headers,
      // body: JSON.stringify(),
    })
      .then(response => response.json())
      .then(response => {
        // console.log(response.data);
        // console.log(item._id);
        // console.log(response.data.similar_item[0]);
        setSimilaritems(response.data.similar_item);
        setSameData(response.data.same_data);
      })
      .catch(err => {
        console.log(err);
      });
  };

  const onGallery = async () => {
    ImagePicker.openPicker({
      compressImageMaxWidth: 300,
      compressImageMaxHeight: 400,
      cropping: true,
      compressImageQuality: 1,
    }).then(async image => {
      imageUpload(image.path);
      setImageP(image.path);
    });
  };

  const imageUpload = async (imagePath: string) => {
    const uriSplit = imagePath.split('.');
    const fileType = uriSplit[uriSplit.length - 1];
    const token = await AsyncStorage.getItem('token');
    const formData = new FormData();
    formData.append('file', {
      // type: 'image/jpg',
      uri: imagePath,
      // uri: imagePath.replace(/file:\/\//g, ''),
      name: imagePath.split('/').pop(),
      type: `image/${fileType}`,
      mimeType: `image/${fileType}`,
    });
    const imageUploadRes = await fetch(
      `${BASE_URL}/api/v1/item-image`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data',
          Accept: 'text/plain',
          Authorization: 'Bearer ' + token,
        },
        body: formData,
      },
    );
    // console.log(imageUploadRes);
    if (imageUploadRes.ok) {
      const uploadImageUrl = await imageUploadRes.json();
      setImageUrl(uploadImageUrl.item_image);
      setImageId(uploadImageUrl.image_id);
      // console.log(uploadImageUrl.item_image);
      // console.log(item._id);
      const editImageRes = await fetch(
        `${BASE_URL}/api/v1/edit-item/${item._id}`,
        {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/plain',
            Authorization: 'Bearer ' + token,
          },
          body: JSON.stringify({
            item_image: uploadImageUrl.item_image,
            image_id: uploadImageUrl.image_id,
          }),
        },
      );
      // console.log('edit log');
      // console.log(editImageRes);
      if (editImageRes.ok) {
        // navigation.navigate('Home');
      } else {
        setError('An error occur');
        setImageP(null);
      }
      // imageData.append('item_image', uploadImageUrl.item_image);
    }
  };

  const submitHandler = async () => {
    setIsEditDesc(false);
    const token = await AsyncStorage.getItem('token');
    const editImageRes = await fetch(
      `${BASE_URL}/api/v1/edit-item/${item._id}`,
      {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/plain',
          Authorization: 'Bearer ' + token,
        },
        body: JSON.stringify({
          item_desc: editDesc,
        }),
      },
    );
    if (editImageRes.ok) {
      // navigation.navigate('Home');
    } else {
      setError('An error occur');
      // setImageP(null);
    }
  };

  const submitHandlerTitle = async () => {
    setIsEditTitle(false);
    const token = await AsyncStorage.getItem('token');
    const editImageRes = await fetch(
      `${BASE_URL}/api/v1/edit-item/${item._id}`,
      {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/plain',
          Authorization: 'Bearer ' + token,
        },
        body: JSON.stringify({
          item_title: editTitle,
        }),
      },
    );
    if (editImageRes.ok) {
      // navigation.navigate('Home');
      setIsEditTitle(false);
    } else {
      setError('An error occur');
    }
  };
  const getUser = async () => {
    const token = await AsyncStorage.getItem('token');
    var url =
      `${BASE_URL}/api/v1/get-user`;
    var headers = {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + token,
    };
    fetch(url, {
      method: 'GET',
      headers: headers,
    })
      .then(response => response.json())
      .then(response => {
        setUser(response.data.role);
      })
      .catch(err => {
        console.log(err);
      });
  };
  const renderSimilar = () => {
    // console.log(legs.length);
    return similarItems.map((data, index) => {
      return (
        <OnTouch onPress={() => handleClick(data.url)} key={index}>
          <ItemUserTier key={index}>
            <ItemUserTierImage source={{uri: data.link}} />
            <ItemUserTierTitleText>{data.title}</ItemUserTierTitleText>
          </ItemUserTier>
        </OnTouch>
      );
    });
  };
  const renderSameData = () => {
    // console.log(legs.length);
    return sameData.map((data, index) => {
      return (
        <OnTouch onPress={() => handleClick(data.url)} key={index}>
          <ItemUserTier key={index}>
            <ItemUserTierImage source={{uri: data.link}} />
            <ItemUserTierTitleText>{data.title}</ItemUserTierTitleText>
          </ItemUserTier>
        </OnTouch>
      );
    });
  };
  useEffect(() => {
    setEditDesc(item.item_desc);
    setEditTitle(item.item_title);
    getSimilarItems();
    getUser();
    // getDaily();
    if (error) {
      Alert.alert('An Error Occurred!', error, [{text: 'Okay'}]);
    }
  }, [error, item]);
  return (
    <SafeView>
      <BackgroundTop />
      <Wrapper>
        <HeaderContainer navigation={navigation} />
        <Spacer />
      </Wrapper>
      <OnScroll>
        <ItemDetailContainer>
          <ItemDetailContainerPadding>
            <ItemTitleContainer>
              <ItemTier>
                <TierLevelIcon>
                  <TierLevelText>I</TierLevelText>
                  <TierImage source={JSON.parse(UserTier)} />
                </TierLevelIcon>
                <UserTierText>User Tier</UserTierText>
              </ItemTier>
              <ItemTier>
                {isEditTitle === false ? (
                  <OnTouch
                    onPress={() => {
                      setIsEditTitle(true);
                    }}>
                    <TierImage source={JSON.parse(EditIcon)} />
                  </OnTouch>
                ) : null}
                {isEditTitle === false ? (
                  <UserTierText numberOfLines={1} ellipsizeMode="tail">
                    {editTitle}
                  </UserTierText>
                ) : null}
                {isEditTitle ? (
                  <TextInput
                    autoCapitalize="none"
                    autoComplete="off"
                    returnKeyType="done"
                    multiline={true}
                    value={editTitle}
                    onChangeText={text => setEditTitle(text)}
                    style={{marginRight: 40, width: 100}}
                  />
                ) : null}
                {isEditTitle ? (
                  <OnTouch
                    onPress={() => {
                      submitHandlerTitle();
                    }}>
                    {/* <ItemDescriptionEditImage source={JSON.parse(CheckIcon)} /> */}
                    <TierImage source={JSON.parse(CheckIcon)} />
                  </OnTouch>
                ) : null}
              </ItemTier>
              <ItemGoBack>
                <OnTouch onPress={() => navigation.navigate('Home')}>
                  <ItemGoBackImage source={JSON.parse(X)} />
                </OnTouch>
              </ItemGoBack>
            </ItemTitleContainer>
          </ItemDetailContainerPadding>
          <ItemDetailContainerPadding>
            <ItemDescription>
              <OnTouchEditImage onPress={onGallery}>
                <ItemEditImage source={JSON.parse(EditImage)} />
              </OnTouchEditImage>
              {imageP !== null ? (
                <ItemDescriptionImage source={{uri: imageP}} />
              ) : (
                <ItemDescriptionImage
                  source={
                    item.item_image
                      ? {uri: item.item_image}
                      : JSON.parse(SampleImage)
                  }
                />
              )}

              <ItemDescriptionDetail>
                <ItemDescriptionDetailHeader>
                  <ItemAddedDateText>
                    Add Date | {Moment(item.createdAt).format('Do MMM YYYY')}
                  </ItemAddedDateText>
                  <OnTouch
                    onPress={() => {
                      if (isEditDesc === false) {
                        setIsEditDesc(true);
                      } else {
                        submitHandler();
                      }
                    }}>
                    <ItemDescriptionEditImage
                      source={
                        isEditDesc ? JSON.parse(CheckIcon) : JSON.parse(Edit)
                      }
                    />
                  </OnTouch>
                </ItemDescriptionDetailHeader>
                {isEditDesc ? (
                  <TextInput
                    autoCapitalize="none"
                    autoComplete="off"
                    returnKeyType="next"
                    multiline={true}
                    value={editDesc}
                    onChangeText={text => setEditDesc(text)}
                    // placeholder="Password"
                  />
                ) : (
                  <ItemDescriptionText>{editDesc}</ItemDescriptionText>
                )}
              </ItemDescriptionDetail>
            </ItemDescription>
          </ItemDetailContainerPadding>
          {user === 'tier-II' || user === 'tier-III' ? (
            <ItemDetailContainerPadding>
              <ItemDescriptionDetailHeader>
                <ItemTier>
                  <TierLevelIcon>
                    <TierLevelText>II</TierLevelText>
                    <TierImage source={JSON.parse(UserTier)} />
                  </TierLevelIcon>
                  <UserTierText>User Tier</UserTierText>
                </ItemTier>
                <ItemTierDivider />
              </ItemDescriptionDetailHeader>
              {/* <ScrollViwWrapper>
            <OnScrollHorizontal horizontal>
              {renderSimilar()}
            </OnScrollHorizontal>
          </ScrollViwWrapper>
          <Spacer /> */}
              <ChartContainer>
                <Chart
                  onItem={dataItem}
                  onCat={dataCat}
                  fullitem={item}
                  item_id={item._id}
                  user={user}
                  navigation={navigation}
                />
              </ChartContainer>
              <Spacer />
              <ItemDescriptionDetailHeader>
                <ItemTierDivider />
                <Text
                  style={{
                    fontWeight: 'bold',
                    marginHorizontal: 10,
                    color: '#000',
                  }}>
                  Matching Items
                </Text>
                <ItemTierDivider />
              </ItemDescriptionDetailHeader>
              <Spacer />
              {sameData.length === 0 ? (
                <NoItem>No item available</NoItem>
              ) : (
                <ScrollViwWrapper>
                  <OnScrollHorizontal horizontal>
                    {renderSameData()}
                  </OnScrollHorizontal>
                </ScrollViwWrapper>
              )}
            </ItemDetailContainerPadding>
          ) : (
            <View
              style={{
                flex: 1,
                // position: 'absolute',
                left: 0,
                top: 0,
                opacity: 0.1,
                padding: 10,
                backgroundColor: 'black',
                width: getWidth,
              }}>
              <OnTouch
                onPress={() => {
                  navigation.navigate('AccountMgmt');
                }}>
                <ItemDescriptionDetailHeader>
                  <ItemTier>
                    <TierLevelIcon>
                      <TierLevelText>II</TierLevelText>
                      <TierImage source={JSON.parse(UserTier)} />
                    </TierLevelIcon>
                    <UserTierText>User Tier</UserTierText>
                  </ItemTier>
                  <ItemTierDivider />
                </ItemDescriptionDetailHeader>
                {/* <ScrollViwWrapper>
            <OnScrollHorizontal horizontal>
              {renderSimilar()}
            </OnScrollHorizontal>
          </ScrollViwWrapper>
          <Spacer /> */}
                <ChartContainer>
                  <Chart
                    onItem={dataItem}
                    fullitem={item}
                    onCat={dataCat}
                    item_id={item._id}
                    user={user}
                    navigation={navigation}
                  />
                </ChartContainer>
                <Spacer />
                <ItemDescriptionDetailHeader>
                  <ItemTierDivider />
                  <Text
                    style={{
                      fontWeight: 'bold',
                      marginHorizontal: 10,
                      color: '#000',
                    }}>
                    Matching Items
                  </Text>
                  <ItemTierDivider />
                </ItemDescriptionDetailHeader>
                <Spacer />
                {sameData.length === 0 ? (
                  <NoItem>No item available</NoItem>
                ) : (
                  <ScrollViwWrapper>
                    <OnScrollHorizontal horizontal>
                      {renderSameData()}
                    </OnScrollHorizontal>
                  </ScrollViwWrapper>
                )}
              </OnTouch>
            </View>
          )}

          <Spacer />
          {user === 'tier-III' ? (
            <ItemDetailContainerPadding>
              <ItemDescriptionDetailHeader>
                <ItemTier>
                  <TierLevelIcon>
                    <TierLevelText>III</TierLevelText>
                    <TierImage source={JSON.parse(UserTier)} />
                  </TierLevelIcon>
                  <UserTierText>User Tier</UserTierText>
                </ItemTier>
                <ItemTierDivider />
              </ItemDescriptionDetailHeader>
              <Spacer />
              <ItemDescriptionDetailHeader>
                <ItemTierDivider />
                <Text
                  style={{
                    fontWeight: 'bold',
                    marginHorizontal: 10,
                    color: '#000',
                  }}>
                  Related Items
                </Text>
                <ItemTierDivider />
              </ItemDescriptionDetailHeader>
              <Spacer />
              {sameData.length === 0 ? (
                <NoItem>No similar item</NoItem>
              ) : (
                <ScrollViwWrapper>
                  <OnScrollHorizontal horizontal>
                    {renderSimilar()}
                  </OnScrollHorizontal>
                </ScrollViwWrapper>
              )}
            </ItemDetailContainerPadding>
          ) : (
            <View
              style={{
                flex: 1,
                // position: 'absolute',
                left: 0,
                top: 0,
                opacity: 0.1,
                padding: 10,
                backgroundColor: 'black',
                width: getWidth,
              }}>
              <OnTouch
                onPress={() => {
                  navigation.navigate('AccountMgmt');
                }}>
                <ItemDescriptionDetailHeader>
                  <ItemTier>
                    <TierLevelIcon>
                      <TierLevelText>III</TierLevelText>
                      <TierImage source={JSON.parse(UserTier)} />
                    </TierLevelIcon>
                    <UserTierText>User Tier</UserTierText>
                  </ItemTier>
                  <ItemTierDivider />
                </ItemDescriptionDetailHeader>
                <Spacer />
                <ItemDescriptionDetailHeader>
                  <ItemTierDivider />
                  <Text
                    style={{
                      fontWeight: 'bold',
                      marginHorizontal: 10,
                      color: '#000',
                    }}>
                    Related Items
                  </Text>
                  <ItemTierDivider />
                </ItemDescriptionDetailHeader>
                <Spacer />
                {sameData.length === 0 ? (
                  <NoItem>No similar item</NoItem>
                ) : (
                  <ScrollViwWrapper>
                    <OnScrollHorizontal horizontal>
                      {renderSimilar()}
                    </OnScrollHorizontal>
                  </ScrollViwWrapper>
                )}
              </OnTouch>
            </View>
          )}
          <Spacer />
          {/* <ItemDescriptionDetailHeader>
            <ItemTier>
              <TierLevelIcon>
                <TierLevelText>III</TierLevelText>
                <TierImage source={JSON.parse(UserTier)} />
              </TierLevelIcon>
              <UserTierText>User Tier</UserTierText>
            </ItemTier>
            <ItemTierDivider />
          </ItemDescriptionDetailHeader>
          <ScrollViwWrapper>
            <OnScrollHorizontal horizontal>
              <ItemUserTierVariant>
                <ItemUserTierImage source={JSON.parse(SampleImage)} />
                <ItemUserTierTitleText>Lorem ipsum dolor</ItemUserTierTitleText>
              </ItemUserTierVariant>
              <ItemUserTierVariant>
                <ItemUserTierImage source={JSON.parse(SampleImage)} />
                <ItemUserTierTitleText>User Tier</ItemUserTierTitleText>
              </ItemUserTierVariant>
              <ItemUserTierVariant>
                <ItemUserTierImage source={JSON.parse(SampleImage)} />
                <ItemUserTierTitleText>User Tier</ItemUserTierTitleText>
              </ItemUserTierVariant>
            </OnScrollHorizontal>
          </ScrollViwWrapper> */}
        </ItemDetailContainer>
      </OnScroll>
      <BackgroundBottom />
    </SafeView>
  );
};

export default JobDetailScreen;
