/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable react-native/no-inline-styles */
import {
  AddItemContainer,
  BackButton,
  ButtonText,
  FormGroup,
  ImageUpload,
  ImageUploadBox,
  ImageUploadContainer,
  ImageUploadText,
  Input,
  InputLabel,
  OnScroll,
  ProfileTitle,
  ProfileTitleText,
  SubmitButton,
} from '../components/add.job.item.styles';
import {
  BackgroundBottom,
  BackgroundTop,
} from '../../../components/backgrounds/backgroundImage';
import {Dimensions, Platform, Text, TouchableOpacity, View} from 'react-native';
import {ErrorText, OnTouch} from '../../auth/components/login.styles';
/* eslint-disable @typescript-eslint/no-unused-vars */
import React, {FC, useEffect, useState} from 'react';
import {useDispatch, useSelector} from 'react-redux';

import AsyncStorage from '@react-native-async-storage/async-storage';
import HeaderContainer from '../../../components/header-components/header.component';
import ImagePicker from 'react-native-image-crop-picker';
import {Loading} from './../../auth/components/register.styles';
import {RootState} from '../../../store/store';
import {SafeView} from '../../../components/utility/safeAreaView.component';
import SelectDropdown from 'react-native-select-dropdown';
import {Spacer} from '../../../components/utility/spacer';
import UploadIcon from '../../../../assets/upload.png';
import {Wrapper} from '../../../components/utility/wrapperContainer';
import X from '../../../../assets/x.png';
import {addItemStart} from '../../../services/items/item.action';
import apiInstance from '../../../axios';
import {useNavigation} from '@react-navigation/native';
import { BASE_URL } from '../../../config';

interface Props {
  navigation: any;
  item?: null;
}
const width = Dimensions.get('window').width;
const AddJobItem: FC<Props> = ({navigation}) => {
  const nav = useNavigation();
  const [title, setTitle] = useState('');
  const [category, setCategory] = useState('');
  const [photo, setPhoto] = useState<string | null>(null);
  const [categories, setCategories] = useState([]);
  const [keyword, setKeyword] = useState('');
  const [description, setDescription] = useState('');
  const [imageUrl, setImageUrl] = useState(null);
  const [imageId, setImageId] = useState(null);
  const [isNewCat, setIsNewCat] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [item_id, setItemId] = useState('');

  // const {error, isLoading} = useSelector((state: RootState) => state.items);
  // Lorem, ipsum, dolor,
  // Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed risus at elementum nunc sapien gravida fermentum amet semper. Neque, vel, erat a imperdiet.

  const dispatch = useDispatch();

  const onGallery = async () => {
    ImagePicker.openPicker({
      compressImageMaxWidth: 300,
      compressImageMaxHeight: 400,
      cropping: true,
      compressImageQuality: 1,
    }).then(async image => {
      console.log('selected Image', image);
      console.log(
        'androd',
        Platform.OS === 'android'
          ? image.path.replace(/file:\/\//g, '')
          : image.path,
      );
      imageUpload(image.path);
      setPhoto(image.path);
    });
  };

  const imageUpload = async (imagePath: string) => {
    const imageData = new FormData();
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
    console.log(imagePath.split('/').pop());
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
    console.log(imageUploadRes);
    if (imageUploadRes.ok) {
      const uploadImageUrl = await imageUploadRes.json();
      setImageUrl(uploadImageUrl.item_image);
      setImageId(uploadImageUrl.image_id);
      console.log(uploadImageUrl.item_image);
    }
  };
  const getCategories = async () => {
    const token = await AsyncStorage.getItem('token');
    var apiUrl =
      `${BASE_URL}/api/v1/get-category`;

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
        setCategories(response.data);
      })
      .catch(erro => {
        console.log('error');
      });
  };
  const addCategory = async () => {
    const token = await AsyncStorage.getItem('token');
    var apiUrl =
      `${BASE_URL}/api/v1/add-category`;

    var headers = {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + token,
    };
    const res = await fetch(apiUrl, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({data: category.trim()}),
    });
    console.log(res);
    if (res.ok) {
      // console.log('category');
      console.log(res);
    }
  };
  const scrapItem = async (id: any) => {
    const token = await AsyncStorage.getItem('token');
    var apiUrl =
      `${BASE_URL}/api/v1/scrap-item`;
    console.log('item id');
    console.log(id);
    var headers = {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + token,
    };
    const res = await fetch(apiUrl, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({
        item_category: category,
        item_keyword: keyword,
        item_id: id,
      }),
    });
    if (res.ok) {
      console.log(res);
    }
  };
  const submitHandle = async () => {
    if (isNewCat) {
      addCategory();
      setIsLoading(true);
      const token = await AsyncStorage.getItem('token');
      var apiUrl =
        `${BASE_URL}/api/v1/add-item`;

      var headers = {
        Accept: 'application/json',
        'Content-Type': 'application/json',
        Authorization: 'Bearer ' + token,
      };
      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
          item_title: title,
          // item_desc: description,
          item_keywords: keyword,
          item_category: category,
          item_image: imageUrl,
          image_id: imageId,
        }),
      });
      if (res.ok) {
        const resRes = await res.json();
        console.log(resRes);
        scrapItem(resRes.data._id);
        setIsLoading(false);
        setTitle('');
        // setDescription('');
        setCategory('');
        setImageUrl(null);
        setKeyword('');
        setImageId(null);
        setPhoto('');
        setError('');
        navigation.navigate('Home');
        getCategories();
      }
      if (!res.ok) {
        const errRes = await res.json();
        console.log(errRes);
        setError(errRes.message);
        setIsLoading(false);
      }
    } else {
      setIsLoading(true);
      const token = await AsyncStorage.getItem('token');
      var apiUrl =
        `${BASE_URL}/api/v1/add-item`;

      var headers = {
        Accept: 'application/json',
        'Content-Type': 'application/json',
        Authorization: 'Bearer ' + token,
      };
      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
          item_title: title,
          // item_desc: description,
          item_keywords: keyword,
          item_category: category,
          item_image: imageUrl,
          image_id: imageId,
        }),
      });
      if (res.ok) {
        const resRes = await res.json();
        console.log(resRes);
        scrapItem(resRes.data._id);
        setIsLoading(false);
        setTitle('');
        // setDescription('');
        // setCategory('');
        setImageUrl(null);
        setKeyword('');
        setImageId(null);
        setPhoto('');
        setError('');
        navigation.goBack();
        getCategories();
      }
      if (!res.ok) {
        const errRes = await res.json();
        console.log(errRes);
        setError(errRes.message);
        setIsLoading(false);
      }
    }

    // }
  };
  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      getCategories();
    });
    return unsubscribe;
    // if (error !== false) {
    //   navigation.navigate('Home');
    // }
  }, [navigation]);
  return (
    <SafeView>
      <BackgroundTop />

      <Wrapper>
        <HeaderContainer navigation={navigation} />
      </Wrapper>
      <Spacer />
      <OnScroll vertical={true}>
        <AddItemContainer>
          <ProfileTitle>
            <ProfileTitleText>Add Item</ProfileTitleText>
            <OnTouch onPress={() => navigation.goBack()}>
              <BackButton source={JSON.parse(X)} />
            </OnTouch>
          </ProfileTitle>
          {error !== null ? <ErrorText>{error}</ErrorText> : null}
          <ImageUploadContainer>
            <ImageUploadBox>
              <OnTouch onPress={onGallery}>
                <ImageUpload
                  source={
                    photo !== null ? {uri: photo} : JSON.parse(UploadIcon)
                  }
                />
              </OnTouch>
            </ImageUploadBox>
            <ImageUploadText>Upload Image of Item</ImageUploadText>
          </ImageUploadContainer>
          <Spacer />
          <FormGroup>
            <InputLabel>Title</InputLabel>
            <Input
              value={title}
              placeholder="Enter Name of Item"
              onChangeText={(text: React.SetStateAction<string>) => {
                setTitle(text);
                setKeyword(text);
              }}
            />
          </FormGroup>
          <Spacer />
          <FormGroup>
            <InputLabel>Category</InputLabel>
            {isNewCat ? (
              <>
                <Input
                  value={category}
                  placeholder="Enter Item Category"
                  onChangeText={(e: React.SetStateAction<string>) =>
                    setCategory(e)
                  }
                />
                <TouchableOpacity onPress={() => setIsNewCat(false)}>
                  <Text
                    style={{color: 'yellow', fontSize: 14, fontWeight: '600'}}>
                    Select Category instead
                  </Text>
                </TouchableOpacity>
              </>
            ) : (
              <View
                style={{
                  flexDirection: 'row',
                  alignContent: 'center',
                  alignItems: 'center',
                }}>
                <SelectDropdown
                  data={categories.sort()}
                  defaultButtonText="Select Category"
                  buttonTextStyle={{fontSize: 12, color: '#3CB474'}}
                  buttonStyle={{
                    // width: width * 0.85,
                    // backgroundColor: 'grey',
                    backgroundColor: 'rgba(255, 255, 255, 0.72)',
                    borderColor: '#3CB474',
                    borderRadius: 2,
                    marginLeft: 20,
                  }}
                  onSelect={(selectedItem, index) => {
                    setCategory(selectedItem);
                  }}
                  buttonTextAfterSelection={(selectedItem, index) => {
                    return selectedItem;
                  }}
                  rowTextForSelection={(item, index) => {
                    return item.toLowerCase();
                  }}
                />
                <TouchableOpacity onPress={() => setIsNewCat(true)}>
                  <Text
                    style={{color: 'yellow', fontSize: 14, fontWeight: '600'}}>
                    Not here add new
                  </Text>
                </TouchableOpacity>
              </View>
            )}
          </FormGroup>
          <Spacer />
          <FormGroup>
            <InputLabel>Keywords</InputLabel>
            <Input
              value={keyword}
              placeholder="Keyword"
              onChangeText={(text: React.SetStateAction<string>) =>
                setKeyword(text)
              }
            />
          </FormGroup>
          <Spacer />
          {/* <FormGroup>
            <InputLabel>Description</InputLabel>
            <Input
              value={description}
              placeholder="Description"
              multiline={true}
              onChangeText={(text: React.SetStateAction<string>) =>
                setDescription(text)
              }
            />
          </FormGroup>
          <Spacer /> */}
          <OnTouch onPress={submitHandle}>
            <SubmitButton>
              {isLoading === true ? (
                <Loading size={18} color="#fff" />
              ) : (
                <ButtonText>Save</ButtonText>
              )}
            </SubmitButton>
          </OnTouch>
        </AddItemContainer>
      </OnScroll>
      <BackgroundBottom />
    </SafeView>
  );
};

export default AddJobItem;
