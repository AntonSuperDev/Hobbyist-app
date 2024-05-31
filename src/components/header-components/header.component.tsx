import {
  Header,
  Logo,
  LogoImage,
  OnTouch,
  UserImage,
  UserNav,
} from './header.styles';
import React, {FC} from 'react';

import LogoIcon from '../../../assets/logo.png';
import UserIcon from '../../../assets/user-icon.png';

// import UserItem from '../../../assets/user-items.png';

interface Props {
  navigation: any;
}
const HeaderContainer: FC<Props> = ({navigation}) => (
  <Header>
    <Logo>
      <LogoImage source={JSON.parse(LogoIcon)} />
    </Logo>
    <UserNav>
      {/* <OnTouch onPress={() => null}>
        <UserImage source={JSON.parse(UserItem)} />
      </OnTouch> */}
      <OnTouch onPress={() => navigation.navigate('Profile')}>
        <UserImage source={JSON.parse(UserIcon)} />
      </OnTouch>
    </UserNav>
  </Header>
);

export default HeaderContainer;
