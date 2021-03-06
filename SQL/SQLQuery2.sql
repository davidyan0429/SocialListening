/****** Script for SelectTopNRows command from SSMS  ******/
SELECT TOP (1000) [id]
      ,[_projectID]
      ,[_id]
      ,[_platform]
      ,[_publishedAt]
      ,[_topic]
      ,[_url]
      ,[_content]
      ,[_source]
      ,[_commonSentiment]
      ,[_score]
      ,[_viewCount]
      ,[_likeCount]
      ,[_commentCount]
      ,[_repostCount]
      ,[_interactCount]
      ,[_haslink]
      ,[_isOriginal]
      ,[_postFrom]
      ,[_nickName]
      ,[_uid]
      ,[_profileImageUrl]
      ,[_gender]
      ,[_province]
      ,[_city]
      ,[_description]
      ,[_friendCount]
      ,[_followerCount]
      ,[_statusCount]
      ,[_favouriteCount]
      ,[_biFollowerCount]
      ,[_createdAt]
      ,[_verified]
      ,[_verifiedType]
      ,[_verifiedReason]
      ,[_ugc]
      ,[_pbw]
      ,[_originalID]
      ,[_originalPublishedAt]
      ,[_originalContent]
      ,[_originalSource]
      ,[_originalCommentCount]
      ,[_originalRepostCount]
      ,[_originalLikeCount]
      ,[_originalPostFrom]
      ,[_originalUID]
      ,[_originalNickName]
      ,[_originalVerified]
      ,[_isDeleted]
      ,[_spam]
      ,[_images]
      ,[_originalImages]
      ,[_rule]
      ,[_dataTag]
      ,[_imageTag]
      ,[_videoContent]
      ,[_rewardCount]
      ,[_isReward]
      ,[_title]
      ,[_indexid]
      ,[_author]
      ,[_digest]
      ,[_userName]
      ,[_biz]
      ,[_account]
      ,[_codeImageUrl]
      ,[_openId]
      ,[_originalUrl]
      ,[_channel]
      ,[_secondChannel]
      ,[_thirdChannel]
      ,[_floor]
      ,[_isComment]
      ,[_isEssence]
      ,[_userUrl]
      ,[_userLevel]
  FROM [dbo].[rawdata]
  ORDER BY [id] desc