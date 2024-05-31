/* eslint-disable react-native/no-inline-styles */
/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable react-hooks/exhaustive-deps */

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
} from '../components/home.styles';
import {BackgroundTop, Loading} from '../../auth/components/login.styles';
import {
  Dimensions,
  ListRenderItem,
  RefreshControl,
  ScrollView,
  Text,
  View,
} from 'react-native';
import React, {FC, useEffect} from 'react';
import {useDispatch, useSelector} from 'react-redux';

import ArrowDown from '../../../../assets/chevron-down.png';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {BackgroundBottom} from '../../auth/components/login.styles';
import FlatlistItem from '../components/FlatlistItem';
import HeaderContainer from '../../../components/header-components/header.component';
import {RootState} from '../../../store/store';
import {SafeView} from '../../../components/utility/safeAreaView.component';
import {SectionDivider} from '../../account-mgmt/components/account.mgmt.styles';
import SelectDropdown from 'react-native-select-dropdown';
import {Spacer} from '../../../components/utility/spacer';
import {getItemListStart} from '../../../services/items/item.action';
import { BASE_URL } from '../../../config';

// import {View} from 'react-native';

const getWidth = Dimensions.get('window').width;
const getHeight = Dimensions.get('window').height;

interface Props {
  id: string;
  title: string;
  item: {title: string};
  navigation: any;
}

export type item = {
  _id: string;
  item_title: string;
  item_image: string;
  item_keywords: string;
  item_desc: string;
};
const wait = (timeout: number | undefined) => {
  return new Promise(resolve => setTimeout(resolve, timeout));
};

const HomeScreen: FC<Props> = ({navigation}) => {
  const dispatch = useDispatch();
  const {itemList, isLoading} = useSelector((state: RootState) => state.items);
  const [refreshing, setRefreshing] = React.useState(false);
  const [items, setItems] = React.useState([]);
  const [days, setDays] = React.useState(10);
  const [page, setPage] = React.useState(1);
  const [average, setAverage] = React.useState();
  const options = ['Lifetime', '10 Days', '1 Month', '6 Months', '1 Year'];
  const [user, setUser] = React.useState('tier-I');

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    wait(2000).then(() => {
      setItems([]);
      getItems();
      setRefreshing(false);
    });
  }, []);
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
        // response.data.data.
        // console.log('items ' + response.data.length);
        if (response.data !== undefined || null || response.data.length !== 0) {
          setItems(response.data);
          const initItems = response.data;
          const sumItems = initItems.reduce((a, c) => {
            return a + (c.average === '' ? 0 : c.average);
          }, 0);
          setAverage(sumItems.toFixed(2));
        }
      })
      .catch(err => {
        console.log(err);
      });
  };
  // const fetMoreItems = () => {
  //   if (items.isListEnd)
  // };
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
        // console.log(response);
        setUser(response.data.role);
      })
      .catch(err => {
        console.log(err);
      });
  };
  useEffect(() => {
    // getItems();
    const unsubscribe = navigation.addListener('focus', () => {
      getItems();
      getUser();
    });
    return unsubscribe;
  }, [navigation]);

  const renderItem: ListRenderItem<item> | null | undefined = ({
    item,
    index,
  }) => (
    <FlatlistItem role={user} item={item} navigation={navigation} idx={index} />
  );

  return (
    <SafeView>
      <BackgroundTop />
      <Wrapper>
        <HeaderContainer navigation={navigation} />
      </Wrapper>
      <SearchContainer>
        <SearchBox>
          {/* <SearchInput placeholder="TODAYS CATALOGE VALUE" /> */}
          <Text
            style={{
              backgroundColor: 'rgba(255, 255, 255, 0.56)',
              width: getWidth * 0.9,
              height: 44,
              textAlign: 'center',
              paddingVertical: 15,
              fontWeight: 'bold',
              fontSize: 13,
              color: '#096C5C',
            }}>
            TODAYS CATALOGUE VALUE {average}
          </Text>
        </SearchBox>
        {/* <InputDropdown>
          <IconDivider />
          <SelectDropdown
            data={options}
            buttonTextStyle={{fontSize: 12, color: '#096C5C'}}
            buttonStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.56)',
              width: (getWidth - 48) / 3,
              height: 44,
            }}
            defaultValue="Lifetime"
            onSelect={(selectedItem, index) => {
              if (selectedItem === 'Lifetime') {
                setItems([]);
                setDays('400');
                getItems();
              } else if (selectedItem === '10 Days') {
                setItems([]);
                setDays('10');
                getItems();
              } else if (selectedItem === '1 Month') {
                setItems([]);
                setDays('30');
                getItems();
              } else if (selectedItem === '6 Months') {
                setItems([]);
                setDays('180');
                getItems();
              } else if (selectedItem === '1 Year') {
                setItems([]);
                setDays('365');
                getItems();
              } else {
                setItems([]);
                setDays('5');
                getItems();
              }
            }}
            buttonTextAfterSelection={(selectedItem, index) => {
              return selectedItem;
            }}
            rowTextForSelection={(item, index) => {
              return item;
            }}
          />
          <ImageIcon source={JSON.parse(ArrowDown)} />
        </InputDropdown> */}
      </SearchContainer>
      <FlatListWrapper>
        <BackgroundPolygon />
        <View
          style={{
            flexDirection: 'row',
            paddingHorizontal: 15,
            paddingVertical: 10,
            alignItems: 'center',
            justifyContent: 'center',
          }}>
          <Text
            style={{
              width: getWidth * 0.43,
              textAlign: 'center',
              fontWeight: 'bold',
              color: '#000',
            }}>
            Item
          </Text>
          <Text
            style={{
              width: getWidth * 0.43,
              textAlign: 'center',
              fontWeight: 'bold',
              color: '#000',
            }}>
            Daily item average price
          </Text>
          <Text
            style={{
              width: getWidth - (getWidth * 0.43 + getWidth * 0.43 + 20),
              textAlign: 'center',
              fontWeight: 'bold',
              color: '#000',
            }}>
            Item detail
          </Text>
        </View>
        {items.length === 0 ? (
          <>
            <Spacer />
            <NoItem>Items not Found for this account</NoItem>
          </>
        ) : (
          <FlatListView
            data={items}
            renderItem={renderItem}
            keyExtractor={item => item._id}
            ListFooterComponent={<SectionDivider />}
            ListFooterComponentStyle={{height: 570}}
            extraData={itemList}
            // onEndReached={fetMoreItems}
            refreshControl={
              <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
            }
          />
        )}
      </FlatListWrapper>

      <BackgroundBottom />
      <View style={{margin: 50}} />
    </SafeView>
  );
};

export default HomeScreen;
