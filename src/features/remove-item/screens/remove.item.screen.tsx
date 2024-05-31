/* eslint-disable react-native/no-inline-styles */
import {Alert, ListRenderItem, View} from 'react-native';
// import apiInstance from '../../../axios';
// import {getItemListStart} from '../../../services/items/item.action';
// import {removeItemStart} from '../../../services/items/item.action';
import {
  BackButton,
  ProfileTitle,
} from './../../add-job-item/components/add.job.item.styles';
import {
  BackgroundPolygon,
  FlatListView,
  FlatListWrapper,
  IconDivider,
  ImageIcon,
  InputDropdown,
  NoItem,
  SearchBox,
  SearchContainer,
  SearchInput,
  SelectInput,
  Wrapper,
} from '../../Home/components/home.styles';
import React, {FC, useEffect, useState} from 'react';

import ArrowDown from '../../../../assets/chevron-down.png';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {BackgroundBottom} from '../../auth/components/login.styles';
import {BackgroundTop} from '../../auth/components/login.styles';
import FlatlistItemRemove from '../components/FlatlistItem';
import HeaderContainer from '../../../components/header-components/header.component';
import {OnTouch} from '../../../components/header-components/header.styles';
import {ProfileTitleText} from './../../profile/components/profile.styles';
// import {RootState} from '../../../store/store';
import {SafeView} from '../../../components/utility/safeAreaView.component';
import {SectionDivider} from '../../account-mgmt/components/account.mgmt.styles';
import {Spacer} from '../../../components/utility/spacer';
import X from '../../../../assets/x.png';
import { BASE_URL } from '../../../config';

// import {useDispatch, useSelector} from 'react-redux';

type Props = {
  navigation: any;
};
type item = {
  _id: string;
  item_title: string;
  item_image: string;
  item_keywords: string;
  item_desc: string;
};

const RemoveItem: FC<Props> = ({navigation}) => {
  // const [refreshFlatlist, setRefreshFlatList] = useState(false);
  const [items, setItems] = useState([]);

  // const dispatch = useDispatch();
  // const {itemList} = useSelector((state: RootState) => state.items);

  const getItems = async () => {
    const token = await AsyncStorage.getItem('token');
    var url =
      `${BASE_URL}/api/v1/items-list`;
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
        setItems(response.data);
      })
      .catch(err => {
        console.log(err);
      });
  };
  const handleDelete = async (id: string) => {
    const token = await AsyncStorage.getItem('token');
    var url = `${BASE_URL}/api/v1/delete-item/${id}`;
    var headers = {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + token,
    };
    await fetch(url, {
      method: 'DELETE',
      headers: headers,
    }).then(response => {
      if (response.ok) {
        Alert.alert('Alert', 'The item is deleted successfully', [
          {text: 'Okay', onPress: () => navigation.goBack()},
        ]);
      }
    });
    // getItems();
  };
  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      getItems();
    });
    return unsubscribe;
  }, [navigation]);

  // const handleDelete = async (id: string) => {
  //   dispatch(removeItemStart(id));
  //   setRefreshFlatList(!refreshFlatlist);
  // };
  const renderItem: ListRenderItem<item> | null | undefined = ({item}) => (
    <FlatlistItemRemove
      item={item}
      navigation={navigation}
      deleteItem={() => handleDelete(item._id)}
      // onRemove={handleDelete}
    />
  );

  return (
    <SafeView>
      <BackgroundTop />
      <Wrapper>
        <HeaderContainer navigation={navigation} />
      </Wrapper>
      <SearchContainer>
        <SearchBox>
          <SearchInput placeholder="TODAYS CATALOGE VALUE" />
        </SearchBox>
        <InputDropdown>
          <IconDivider />
          <SelectInput placeholder="Lifetime" />
          <ImageIcon source={JSON.parse(ArrowDown)} />
        </InputDropdown>
      </SearchContainer>
      <FlatListWrapper>
        <BackgroundPolygon />
        <View style={{alignItems: 'flex-end', marginRight: 20, marginTop: 10}}>
          <OnTouch onPress={() => navigation.goBack()}>
            <BackButton source={JSON.parse(X)} />
          </OnTouch>
        </View>
        {items.length === 0 ? (
          <Spacer>
            <NoItem>Items not Found for this account</NoItem>
          </Spacer>
        ) : (
          <FlatListView
            data={items}
            renderItem={renderItem}
            keyExtractor={item => item._id}
            ListFooterComponent={<SectionDivider />}
            ListFooterComponentStyle={{height: 300}}
            extraData={items}
          />
        )}
      </FlatListWrapper>
      <BackgroundBottom />
    </SafeView>
  );
};

export default RemoveItem;
