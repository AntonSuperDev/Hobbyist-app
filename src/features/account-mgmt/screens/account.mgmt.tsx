/* eslint-disable prettier/prettier */

import {
  AddItemContainer,
  CurrentButton,
  CurrentButtonText,
  ItemImage,
  ItemTitleContainer,
  SectionDivider,
  SubscribeButtonBlue,
  SubscribeButtonPink,
  TierBoxLeft,
  TierCollector,
  TierCollectorList,
  TierHeader,
  TierImage,
  TierLeftGreenText,
  TierLeftPurpleText,
  TierLeftText,
  TierLevelBlueText,
  TierLevelIcon,
  TierLevelPurpleText,
  TierLevelText,
  TierListItem,
  TierListItemText,
} from '../components/account.mgmt.styles';
import {Modal, NativeBaseProvider} from 'native-base';
import {WebView} from 'react-native-webview';
import {
  BackButton,
  ProfileTitleText,
} from '../../profile/components/profile.styles';
import {
  BackgroundBottom,
  BackgroundBottomLeft,
  BackgroundTop,
} from '../../../components/backgrounds/backgroundImage';
import React, {FC, useState} from 'react';

import {withIAPContext} from 'react-native-iap';
import Check from '../../../../assets/check.png';
import CheckGreen from '../../../../assets/check-green.png';
import CheckPurple from '../../../../assets/check-purple.png';
import GrrenIcon from '../../../../assets/user-tier.png';
import HeaderContainer from '../../../components/header-components/header.component';
import {
  Loading,
  OnTouch,
  TermAndContions,
  TermAndContionsText,
  TermAndContionsTextColor,
} from '../../auth/components/login.styles';
import Pouse from '../../../../assets/pouse.png';
import PurpleIconIcon from '../../../../assets/purple-icon.png';
import {SafeView} from '../../../components/utility/safeAreaView.component';
import {Spacer} from '../../../components/utility/spacer';
import {Wrapper} from '../../../components/utility/wrapperContainer';
import X from '../../../../assets/x.png';
import useInAppPurchase from '../../in-app-pucharse/hooks/useInAppPucharse';
import {Button, Text, View} from 'react-native';
import {ScrollView} from 'react-native-gesture-handler';
import TermsModal from '../components/Modal';

interface Props {
  navigation: any;
}
const AccountMgmt: FC<Props> = ({navigation}) => {
  const {
    setSelectedTier,
    loadingProducts,
    tierAvailable,
    currentLoadingButton,
    restoreSubscription,
    isRestoring,
  } = useInAppPurchase();
  const [showModal, setShowModal] = useState(false);
  return (
    <>
      <ScrollView>
        <SafeView>
          <BackgroundTop />
          <Wrapper>
            <HeaderContainer navigation={navigation} />
          </Wrapper>
          <Spacer />
          <AddItemContainer>
            <ItemTitleContainer>
              <ProfileTitleText>Account Management</ProfileTitleText>
              <OnTouch onPress={() => navigation.goBack()}>
                <BackButton source={JSON.parse(X)} />
              </OnTouch>
            </ItemTitleContainer>
            <SectionDivider />
            <TierCollector>
              <TierHeader>
                <TierBoxLeft>
                  <TierLevelIcon>
                    <TierLevelText>I</TierLevelText>
                    <TierImage source={JSON.parse(GrrenIcon)} />
                  </TierLevelIcon>
                  <TierLeftGreenText>Tier Collector</TierLeftGreenText>
                </TierBoxLeft>
                <OnTouch onPress={() => setSelectedTier(2)}>
                  <CurrentButton>
                    <CurrentButtonText>
                      {tierAvailable === 2 ? 'Current' : 'Free tier'}
                    </CurrentButtonText>
                  </CurrentButton>
                </OnTouch>
              </TierHeader>
            </TierCollector>
            <TierCollectorList>
              <TierListItem>
                <ItemImage source={JSON.parse(CheckGreen)} />
                <TierListItemText>
                  Allows entry of your collectibles into your hobbyist catalogue
                </TierListItemText>
              </TierListItem>
            </TierCollectorList>
            <Spacer />
            <SectionDivider />
            <TierCollector>
              <TierHeader>
                <TierBoxLeft>
                  <TierLevelIcon>
                    <TierLevelBlueText>II</TierLevelBlueText>
                    <TierImage source={JSON.parse(Pouse)} />
                  </TierLevelIcon>
                  <TierLeftText>Tier Collector</TierLeftText>
                </TierBoxLeft>
                <OnTouch
                  onPress={() => setSelectedTier(0)}
                  disabled={
                    loadingProducts ||
                    tierAvailable === 0 ||
                    currentLoadingButton === 0
                  }>
                  <SubscribeButtonBlue>
                    {currentLoadingButton === 0 && (
                      <View
                        style={{
                          display: 'flex',
                          width: '100%',
                          height: '100%',
                          marginTop: 8,
                          flex: 1,
                          justifyContent: 'center',
                          position: 'absolute',
                        }}>
                        <Loading size={18} color="#000" />
                      </View>
                    )}
                    <CurrentButtonText>
                      <Text
                        style={{opacity: currentLoadingButton === 0 ? 0 : 1}}>
                        {tierAvailable === 0 ? 'Current' : '$4.99 / Month'}
                      </Text>
                    </CurrentButtonText>
                  </SubscribeButtonBlue>
                </OnTouch>
              </TierHeader>
            </TierCollector>
            <TierCollectorList>
              <TierListItem>
                <ItemImage source={JSON.parse(Check)} />
                <TierListItemText>
                  Daily pricing of your collectibles and total catalogue value
                </TierListItemText>
              </TierListItem>
              <TierListItem>
                <ItemImage source={JSON.parse(Check)} />
                <TierListItemText>
                  Trending your collectibles value over time so you understand
                  if it’s gaining or losing value (trend line does not start
                  until you’ve entered item in app)
                </TierListItemText>
              </TierListItem>
              <TierListItem>
                <ItemImage source={JSON.parse(Check)} />
                <TierListItemText>
                  Rapid linking to similar collectibles as your catalogue on
                  e-commerce sites
                </TierListItemText>
              </TierListItem>
            </TierCollectorList>
            <Spacer />
            <SectionDivider />
            <TierCollector>
              <TierHeader>
                <TierBoxLeft>
                  <TierLevelIcon>
                    <TierLevelPurpleText>III</TierLevelPurpleText>
                    <TierImage source={JSON.parse(PurpleIconIcon)} />
                  </TierLevelIcon>
                  <TierLeftPurpleText>Tier Collector</TierLeftPurpleText>
                </TierBoxLeft>
                <OnTouch
                  onPress={() => setSelectedTier(1)}
                  disabled={
                    loadingProducts ||
                    tierAvailable === 1 ||
                    currentLoadingButton === 1
                  }>
                  <SubscribeButtonPink>
                    {currentLoadingButton === 1 && (
                      <View
                        style={{
                          display: 'flex',
                          width: '100%',
                          height: '100%',
                          marginTop: 8,
                          flex: 1,
                          justifyContent: 'center',
                          position: 'absolute',
                        }}>
                        <Loading size={18} color="#000" />
                      </View>
                    )}
                    <CurrentButtonText>
                      <Text
                        style={{opacity: currentLoadingButton === 1 ? 0 : 1}}>
                        {tierAvailable === 1 ? 'Current' : '$9.99 / Month'}
                      </Text>
                    </CurrentButtonText>
                  </SubscribeButtonPink>
                </OnTouch>
              </TierHeader>
            </TierCollector>
            <TierCollectorList>
              <TierListItem>
                <ItemImage source={JSON.parse(CheckPurple)} />
                <TierListItemText>
                  Daily pricing of your collectibles category so you can know if
                  your collectible is outpacing its peer collectibles in a
                  category
                </TierListItemText>
              </TierListItem>
              <TierListItem>
                <ItemImage source={JSON.parse(CheckPurple)} />
                <TierListItemText>
                  Trending your collectibles category average price over time
                </TierListItemText>
              </TierListItem>
              <TierListItem>
                <ItemImage source={JSON.parse(CheckPurple)} />
                <TierListItemText>
                  Rapid linking to collectibles on e-commerce sites that are
                  different than your collectible but in the same category for
                  catalogue optimization
                </TierListItemText>
              </TierListItem>
            </TierCollectorList>
            <BackgroundBottomLeft />
            {isRestoring ? (
              <View style={{marginTop: 20, marginBottom: 20}}>
                <Loading />
              </View>
            ) : (
              <Button
                title={'Restore subscription'}
                onPress={restoreSubscription}
              />
            )}
            <Button
              title={'Terms and privacy policy'}
              onPress={() => setShowModal(true)}
              color="gray"
            />
          </AddItemContainer>
          <BackgroundBottom />
        </SafeView>
        <TermsModal showModal={showModal} setShowModal={setShowModal} />
      </ScrollView>
    </>
  );
};

export default withIAPContext(AccountMgmt);
