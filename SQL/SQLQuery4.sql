/****** Script for SelectTopNRows command from SSMS  ******/
SELECT TOP (10000) [id]
      ,[rawId]
      ,[sentiment]
      ,[createAt]
  FROM [dbo].[sentiment]
  ORDER BY id desc